import torch
import torch.nn as nn
from tqdm import tqdm

def train_with_early_stopping(model, train_loader, valid_loader, config, use_onecycle=True):
    """
    Train model with early stopping and OneCycle learning rate scheduler

    Args:
        model: Model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        config: Configuration object
        use_onecycle: Whether to use OneCycleLR scheduler

    Returns:
        best_val_loss: Best validation loss
        best_epoch: Best model epoch
        model: Trained model
    """
    criterion = nn.MSELoss() if not config.direct_train else nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    if use_onecycle:
        # Set up OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate * 5,
            epochs=config.max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            div_factor=5,
            final_div_factor=100,
            anneal_strategy='cos'
        )
    else:
        # Set up ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=config.patience, min_lr=1e-8, verbose=True
        )

    device = config.device
    model.to(device)

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    no_improve_count = 0
    
    for epoch in range(config.max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_vae_loss = 0
        if config.direct_train:
            for i, (x, y, _, _) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                pred = model(x)
                y_indices = torch.argmax(y, dim=1)
                loss = criterion(pred.squeeze(), y_indices)
                
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if use_onecycle:
                    scheduler.step()

                train_loss += loss.item()

        for i, (x, y, _, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y)
 
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if use_onecycle:
                scheduler.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_vae_loss = 0

        with torch.no_grad():
            if config.direct_train:
                for x, y, _, _ in valid_loader:
                    x = x.to(device)
                    y = y.to(device)

                    pred = model(x)
                    y_indices = torch.argmax(y, dim=1)
                    loss = criterion(pred.squeeze(), y_indices)
                    
                    val_loss += loss.item()
            else:
                for x, y, _, _ in valid_loader:
                    x = x.to(device)
                    y = y.to(device)

                    pred = model(x)
                    loss = criterion(pred.squeeze(), y)

                    val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)

        if not use_onecycle:
            scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config.max_epochs}: Train Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}, Learning Rate: {current_lr:.2e}")


        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            print(f"  New best model saved (Validation Loss: {best_val_loss:.6f})")
        else:
            no_improve_count += 1
            print(f"  No improvement: {no_improve_count}/{config.patience}")

        # Check early stopping
        if no_improve_count >= config.patience:
            print(f"Early stopping: No improvement for {config.patience} epochs")
            break

    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model restored (Epoch {best_epoch+1}, Validation Loss: {best_val_loss:.6f})")

    return best_val_loss, best_epoch, model

def train_regime_mamba(model, train_loader, valid_loader, config, save_path=None, progressive_train=0):
    """
    Complete training process for RegimeMamba model

    Args:
        model: Model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        config: Configuration object
        save_path: Model save path (None if not saving)

    Returns:
        model: Trained model
    """
    criterion = nn.MSELoss() if not config.direct_train else nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Set up OneCycleLR scheduler
    total_steps = config.max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 5,  # Maximum learning rate
        total_steps=total_steps,
        pct_start=0.2,                    # First 20% increase learning rate
        div_factor=5,                     # Initial learning rate = max_lr / div_factor
        final_div_factor=100,             # Final learning rate = initial learning rate / final_div_factor
        anneal_strategy='cos'             # Cosine annealing for learning rate
    )

    device = config.device
    model.to(device)

    best_val_loss = float('inf')

    for epoch in range(config.max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_vae_loss = 0
        train_pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} (Training)")

        if config.direct_train:
            for i, (x, y, _, _) in train_pbar:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                y_indices = torch.argmax(y, dim=1)
                loss = criterion(pred.squeeze(), y_indices)
                
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_pbar.set_postfix({"train_loss": train_loss / (i + 1)})

        else:
            for i, (x, y, _, _) in train_pbar:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred.squeeze(), y)
                
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_pbar.set_postfix({"train_loss": train_loss / (i + 1)})

        # Validation phase
        model.eval()
        val_loss = 0
        val_vae_loss = 0

        with torch.no_grad():

            if config.direct_train:
                for i, (x, y, _, _) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)

                    pred = model(x)
                    y_indices = torch.argmax(y, dim=1)
                    loss = criterion(pred.squeeze(), y_indices)

                    val_loss += loss.item()

            else:
                for i, (x, y, _, _) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)

                    pred = model(x)
                    loss = criterion(pred.squeeze(), y)

                    val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}, Learning Rate: {current_lr:.2e}")
        
        # Save model
        if avg_val_loss < best_val_loss and save_path is not None:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            print(f"Model saved (Epoch {epoch+1})")

    # Load best model
    if save_path is not None:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded (Epoch {checkpoint['epoch']+1})")

    return model
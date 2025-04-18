import torch
import torch.nn as nn
from tqdm import tqdm

def train_with_early_stopping(model, train_loader, valid_loader, config, use_onecycle=True, progressive_train=0):
    """
    조기 종료와 OneCycle 학습률 스케줄러를 적용한 모델 훈련

    Args:
        model: 훈련할 모델
        train_loader: 훈련 데이터 로더
        valid_loader: 검증 데이터 로더
        config: 설정 객체
        use_onecycle: OneCycleLR 스케줄러 사용 여부
        progressive_train: 점진적 훈련 단계 (0: 전체, 1: fc_mu, fc_var, decoder 고정, 2: fc_mu, fc_var, decoder만 학습)

    Returns:
        best_val_loss: 최적 검증 손실
        best_epoch: 최적 모델의 에폭
        model: 훈련된 모델
    """
    criterion = nn.MSELoss() if not config.direct_train else nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    if use_onecycle:
        # OneCycleLR 스케줄러 설정
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
        # ReduceLROnPlateau 스케줄러 설정
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=config.patience, min_lr=1e-8, verbose=True
        )

    device = config.device
    model.to(device)

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    no_improve_count = 0

    if progressive_train == 1:
        # 모델의 fc_mu, fc_var, decoder 파라미터를 고정
        # 고정할 파라미터를 집합으로 정의
        frozen_params = {model.fc_mu.weight, model.fc_var.weight, model.decoder[0].weight, model.decoder[1].weight, model.decoder[3].weight,
                         model.decoder[4].weight, model.decoder[6].weight, model.decoder[7].weight, model.decoder[9].weight, model.decoder[10].weight}

        for param in model.parameters():
            param.requires_grad = param not in frozen_params # 고정된 파라미터는 학습하지 않음

    elif progressive_train == 2:
        active_params = {model.fc_mu.weight, model.fc_var.weight, model.decoder[0].weight, model.decoder[1].weight, model.decoder[3].weight,
                         model.decoder[4].weight, model.decoder[6].weight, model.decoder[7].weight, model.decoder[9].weight, model.decoder[10].weight}
        for param in model.parameters():
            param.requires_grad = param in active_params
    
    for epoch in range(config.max_epochs):
        # 훈련 단계
        model.train()
        train_loss = 0
        train_vae_loss = 0
        if config.direct_train:
            for i, (x, y, _, _) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                if config.vae:
                    pred, h, mu, log_var, z, recon = model(x)
                    loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                    if progressive_train == 1:
                        loss = pred_loss
                    elif progressive_train == 2:
                        loss = vae_loss + kl_loss
                else:
                    pred = model(x)
                    y_indices = torch.argmax(y, dim=1)
                    loss = criterion(pred.squeeze(), y_indices)
                
                loss.backward()

                # 그래디언트 클리핑 추가
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if use_onecycle:
                    scheduler.step()

                train_loss += loss.item()
                if config.vae:
                    train_vae_loss += 0.1*vae_loss.item()
                    train_vae_loss += 0.01*kl_loss.item()

        elif config.preprocessed:
            for i, (x, y, _) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                if config.vae:
                    pred, h, mu, log_var, z, recon = model(x)
                    loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                    if progressive_train == 1:
                        loss = pred_loss
                    elif progressive_train == 2:
                        loss = vae_loss + kl_loss
                else:
                    pred = model(x)
                    loss = criterion(pred.squeeze(), y)
                loss.backward()

                # 그래디언트 클리핑 추가
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if use_onecycle:
                    scheduler.step()

                train_loss += loss.item()
                if config.vae:
                    train_vae_loss += 0.1*vae_loss.item()
                    train_vae_loss += 0.01*kl_loss.item()
        else:
            for i, (x, y, _, _) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                if config.vae:
                    pred, h, mu, log_var, z, recon = model(x)
                    loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                    if progressive_train == 1:
                        loss = pred_loss
                    elif progressive_train == 2:
                        loss = vae_loss + kl_loss
                else:
                    pred = model(x)
                    loss = criterion(pred.squeeze(), y)
 
                loss.backward()

                # 그래디언트 클리핑 추가
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if use_onecycle:
                    scheduler.step()

                train_loss += loss.item()
                if config.vae:
                    train_vae_loss += 0.1*vae_loss.item()
                    train_vae_loss += 0.01*kl_loss.item()

        # 검증 단계
        model.eval()
        val_loss = 0
        val_vae_loss = 0

        with torch.no_grad():
            if config.direct_train:
                for x, y, _, _ in valid_loader:
                    x = x.to(device)
                    y = y.to(device)

                    if config.vae:
                        pred, h, mu, log_var, z, recon = model(x)
                        loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                        if progressive_train == 1:
                            loss = pred_loss
                        elif progressive_train == 2:
                            loss = vae_loss + kl_loss
                    else:
                        pred = model(x)
                        y_indices = torch.argmax(y, dim=1)
                        loss = criterion(pred.squeeze(), y_indices)
                    
                    val_loss += loss.item()
                    if config.vae:
                        val_vae_loss += 0.1*vae_loss.item()
                        val_vae_loss += 0.01*kl_loss.item()
            elif config.preprocessed:
                for x, y, _ in valid_loader:
                    x = x.to(device)
                    y = y.to(device)

                    if config.vae:
                        pred, h, mu, log_var, z, recon = model(x)
                        loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                        if progressive_train == 1:
                            loss = pred_loss
                        elif progressive_train == 2:
                            loss = vae_loss + kl_loss
                    else:
                        pred = model(x)
                        loss = criterion(pred.squeeze(), y)

                    val_loss += loss.item()
                    if config.vae:
                        val_vae_loss += 0.1*vae_loss.item()
                        val_vae_loss += 0.01*kl_loss.item()
            else:
                for x, y, _, _ in valid_loader:
                    x = x.to(device)
                    y = y.to(device)

                    if config.vae:
                        pred, h, mu, log_var, z, recon = model(x)
                        loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                        if progressive_train == 1:
                            loss = pred_loss
                        elif progressive_train == 2:
                            loss = vae_loss + kl_loss
                    else:
                        pred = model(x)
                        loss = criterion(pred.squeeze(), y)

                    val_loss += loss.item()
                    if config.vae:
                        val_vae_loss += 0.1*vae_loss.item()
                        val_vae_loss += 0.01*kl_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        if config.vae:
            avg_vae_loss = vae_loss / len(train_loader)
            avg_val_vae_loss = val_vae_loss / len(valid_loader)

        if not use_onecycle:
            scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"에폭 {epoch+1}/{config.max_epochs}: 훈련 손실 = {avg_train_loss:.6f}, 검증 손실 = {avg_val_loss:.6f}, 학습률: {current_lr:.2e}")
        if config.vae:
            print(f"  VAE 손실: {avg_vae_loss:.6f}, 검증 VAE 손실: {avg_val_vae_loss:.6f}")


        # 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
            print(f"  새로운 최적 모델 저장 (검증 손실: {best_val_loss:.6f})")
        else:
            no_improve_count += 1
            print(f"  개선 없음: {no_improve_count}/{config.patience}")

        # 조기 종료 확인
        if no_improve_count >= config.patience:
            print(f"조기 종료: {config.patience} 에폭 동안 개선 없음")
            break

    # 최적 모델 상태로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"최적 모델 복원 (에폭 {best_epoch+1}, 검증 손실: {best_val_loss:.6f})")

    return best_val_loss, best_epoch, model

def train_regime_mamba(model, train_loader, valid_loader, config, save_path=None, progressive_train=0):
    """
    RegimeMamba 모델의 전체 훈련 과정

    Args:
        model: 훈련할 모델
        train_loader: 훈련 데이터 로더
        valid_loader: 검증 데이터 로더
        config: 설정 객체
        save_path: 모델 저장 경로 (None이면 저장하지 않음)
        progressive_train: 점진적 훈련 단계 (0: 전체, 1: fc_mu, fc_var, decoder 고정, 2: fc_mu, fc_var, decoder만 학습)

    Returns:
        model: 훈련된 모델
    """
    criterion = nn.MSELoss() if not config.direct_train else nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # OneCycleLR 스케줄러 설정
    total_steps = config.max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 5,  # 최대 학습률
        total_steps=total_steps,
        pct_start=0.2,                    # 초기 20%는 학습률 증가
        div_factor=5,                     # 초기 학습률 = max_lr / div_factor
        final_div_factor=100,             # 최종 학습률 = 초기 학습률 / final_div_factor
        anneal_strategy='cos'             # 코사인 방식으로 학습률 조정
    )

    device = config.device
    model.to(device)

    if progressive_train == 1:
        # 모델의 fc_mu, fc_var, decoder 파라미터를 고정
        # 고정할 파라미터를 집합으로 정의
        frozen_params = {model.fc_mu.weight, model.fc_var.weight, model.decoder[0].weight, model.decoder[1].weight, model.decoder[3].weight,
                         model.decoder[4].weight, model.decoder[6].weight, model.decoder[7].weight, model.decoder[9].weight, model.decoder[10].weight}

        for param in model.parameters():
            param.requires_grad = param not in frozen_params

    elif progressive_train == 2:
        active_params = {model.fc_mu.weight, model.fc_var.weight, model.decoder[0].weight, model.decoder[1].weight, model.decoder[3].weight,
                         model.decoder[4].weight, model.decoder[6].weight, model.decoder[7].weight, model.decoder[9].weight, model.decoder[10].weight}
        for param in model.parameters():
            param.requires_grad = param in active_params

    best_val_loss = float('inf')

    for epoch in range(config.max_epochs):
        # 훈련 단계
        model.train()
        train_loss = 0
        train_vae_loss = 0
        train_pbar = tqdm(enumerate(train_loader), desc=f"에폭 {epoch+1} (훈련)")

        if config.direct_train:
            for i, (x, y, _, _) in train_pbar:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                if config.vae:
                    pred, h, mu, log_var, z, recon = model(x)
                    loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                    if progressive_train == 1:
                        loss = pred_loss
                    elif progressive_train == 2:
                        loss = vae_loss + kl_loss
                else:
                    pred = model(x)
                    y_indices = torch.argmax(y, dim=1)
                    loss = criterion(pred.squeeze(), y_indices)
                
                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                if config.vae:
                    train_vae_loss += 0.1*vae_loss.item()
                    train_vae_loss += 0.01*kl_loss.item()
                train_pbar.set_postfix({"train_loss": train_loss / (i + 1)})

        elif config.preprocessed:
            for i, (x, y, _) in train_pbar:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                if config.vae:
                    pred, h, mu, log_var, z, recon = model(x)
                    loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                    if progressive_train == 1:
                        loss = pred_loss
                    elif progressive_train == 2:
                        loss = vae_loss + kl_loss
                else:
                    pred = model(x)
                    loss = criterion(pred.squeeze(), y)

                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                if config.vae:
                    train_vae_loss += 0.1*vae_loss.item()
                    train_vae_loss += 0.01*kl_loss.item()
                train_pbar.set_postfix({"train_loss": train_loss / (i + 1)})
        else:
            for i, (x, y, _, _) in train_pbar:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                if config.vae:
                    pred, h, mu, log_var, z, recon = model(x)
                    loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                    if progressive_train == 1:
                        loss = pred_loss
                    elif progressive_train == 2:
                        loss = vae_loss + kl_loss
                else:
                    pred = model(x)
                    loss = criterion(pred.squeeze(), y)
                
                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                if config.vae:
                    train_vae_loss += 0.1*vae_loss.item()
                    train_vae_loss += 0.01*kl_loss.item()
                train_pbar.set_postfix({"train_loss": train_loss / (i + 1)})

        # 검증 단계
        model.eval()
        val_loss = 0
        val_vae_loss = 0

        with torch.no_grad():

            if config.direct_train:
                for i, (x, y, _, _) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)

                    if config.vae:
                        pred, h, mu, log_var, z, recon = model(x)
                        loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                        if progressive_train == 1:
                            loss = pred_loss
                        elif progressive_train == 2:
                            loss = vae_loss + kl_loss
                    else:
                        pred = model(x)
                        y_indices = torch.argmax(y, dim=1)
                        loss = criterion(pred.squeeze(), y_indices)

                    val_loss += loss.item()
                    if config.vae:
                        val_vae_loss += 0.1*vae_loss.item()
                        val_vae_loss += 0.01*kl_loss.item()

            elif config.preprocessed:
                for i, (x, y, _) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)

                    if config.vae:
                        pred, h, mu, log_var, z, recon = model(x)
                        loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                        if progressive_train == 1:
                            loss = pred_loss
                        elif progressive_train == 2:
                            loss = vae_loss + kl_loss
                    else:
                        pred = model(x)
                        loss = criterion(pred.squeeze(), y)

                    val_loss += loss.item()
                    if config.vae:
                        val_vae_loss += 0.1*vae_loss.item()
                        val_vae_loss += 0.01*kl_loss.item()
            else:
                for i, (x, y, _, _) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)

                    if config.vae:
                        pred, h, mu, log_var, z, recon = model(x)
                        loss, vae_loss, kl_loss, pred_loss = model.vae_loss_function(recon, h, mu, log_var, pred, y)
                        if progressive_train == 1:
                            loss = pred_loss
                        elif progressive_train == 2:
                            loss = vae_loss + kl_loss
                    else:
                        pred = model(x)
                        loss = criterion(pred.squeeze(), y)

                    val_loss += loss.item()
                    if config.vae:
                        val_vae_loss += 0.1*vae_loss.item()
                        val_vae_loss += 0.01*kl_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        if config.vae:
            avg_vae_loss = vae_loss / len(train_loader)
            avg_val_vae_loss = val_vae_loss / len(valid_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"에폭 {epoch+1}: 훈련 손실 = {avg_train_loss:.6f}, 검증 손실 = {avg_val_loss:.6f}, 학습률: {current_lr:.2e}")
        if config.vae:
            print(f"  VAE 손실: {avg_vae_loss:.6f}, 검증 VAE 손실: {avg_val_vae_loss:.6f}")

        # 모델 저장
        if avg_val_loss < best_val_loss and save_path is not None:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            print(f"모델 저장됨 (에폭 {epoch+1})")

    # 최적 모델 로드
    if save_path is not None:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"최적 모델 로드됨 (에폭 {checkpoint['epoch']+1})")

    return model
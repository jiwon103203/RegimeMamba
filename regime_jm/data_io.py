"""1) 입력 로드·정제 → 초과수익률 (+ 벤치마크 차감 ``rel_ret``).

``prepare_inputs`` 가 반환하는 DataFrame (DatetimeIndex, 첫 수익률 NaN 행 제거)::

    close       자산 종가
    asset_ret   자산 단순수익률            close_t / close_{t-1} - 1
    rf          원본 무위험금리 (입력 단위 그대로)
    rf_ret      t일 현금 수익률            t-1일에 적용되던 일간 무위험금리
    ret         초과수익률                 asset_ret - rf_ret
    bench_close, bench_ret, rel_ret        (--relative-benchmark 사용 시) rel_ret = asset_ret - bench_ret
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

ENCODINGS = ("utf-8-sig", "cp949", "euc-kr")  # utf-8-sig 는 BOM 없는 utf-8 도 읽는다
RF_UNITS = ("annual_pct", "annual", "daily_pct", "daily")
PERIODS_PER_YEAR = 252

# 정규화된(소문자, 공백/기호 제거) 열 이름 후보. 앞에 있을수록 우선.
DATE_ALIASES = ("date", "dates", "datetime", "timestamp", "time", "날짜", "일자", "일시", "기준일", "기준일자",
                "거래일", "거래일자", "basdt", "trddd")
CLOSE_ALIASES = ("close", "adjclose", "adjustedclose", "closeprice", "price", "last", "pxlast", "종가", "수정종가",
                 "종가지수", "현재가", "지수", "가격", "clpr")
RF_ALIASES = ("rf", "riskfree", "riskfreerate", "rfrate", "무위험금리", "무위험이자율", "무위험", "cd91", "cd91일",
              "cd금리", "cd", "콜금리", "금리", "tbill", "tbill3m", "dgs3mo", "irx", "rate")

_NA_TOKENS = {"", "-", "--", "—", ".", "nan", "NaN", "NAN", "N/A", "n/a", "#N/A", "NA", "null", "None"}


# ---------------------------------------------------------------------------
# 파일 읽기
# ---------------------------------------------------------------------------

def _sniff_delimiter(first_line: str) -> str:
    counts = {sep: first_line.count(sep) for sep in (",", "\t", ";", "|")}
    sep, n = max(counts.items(), key=lambda kv: kv[1])
    return sep if n > 0 else ","


def read_table(path: str, sheet=None) -> pd.DataFrame:
    """csv/tsv/txt/xlsx 를 읽는다. 텍스트 파일은 utf-8 → cp949 → euc-kr 순으로 인코딩을 시도한다."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)

    last_err: Optional[Exception] = None
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                first_line = f.readline()
            sep = "\t" if ext == ".tsv" else _sniff_delimiter(first_line)
            df = pd.read_csv(path, encoding=enc, sep=sep, dtype=str, keep_default_na=False)
            df.attrs["encoding"] = enc
            return df
        except UnicodeDecodeError as e:
            last_err = e
    raise ValueError(f"{path}: 인코딩 {ENCODINGS} 중 어느 것으로도 읽을 수 없습니다") from last_err


def to_numeric(s: pd.Series) -> pd.Series:
    """'1,234.5', '3.5%', '(1.2)', '-' 같은 문자열을 float 로 변환한다 (% 는 기호만 제거)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype("string").str.strip()
    t = t.str.replace(r"[,%\s₩$]", "", regex=True)
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # 회계식 음수
    t = t.mask(t.isin(_NA_TOKENS))
    return pd.to_numeric(t, errors="coerce").astype(float)


def parse_dates(s: pd.Series) -> pd.DatetimeIndex:
    """'2024-01-02', '2024/01/02', '2024.01.02', '20240102', '2024년 1월 2일', 엑셀 serial 을 모두 처리한다."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.DatetimeIndex(s).normalize()
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
        valid = v.dropna()
        if len(valid) and valid.between(19000101, 21001231).all():
            return pd.DatetimeIndex(pd.to_datetime(v.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce"))
        if len(valid) and valid.between(1, 80000).all():
            return pd.DatetimeIndex(pd.to_datetime(v, unit="D", origin="1899-12-30", errors="coerce")).normalize()
    t = s.astype("string").str.strip()
    t = t.str.replace(r"\s*([./년월일])\s*", r"\1", regex=True)  # '2024. 1. 2.' → '2024.1.2.'
    ymd = t.str.fullmatch(r"\d{8}").fillna(False)
    t = t.mask(ymd, t.str.slice(0, 4) + "-" + t.str.slice(4, 6) + "-" + t.str.slice(6, 8))
    t = t.str.replace(r"[./년월]", "-", regex=True).str.replace("일", "", regex=False).str.strip("-")
    return pd.DatetimeIndex(pd.to_datetime(t, errors="coerce", format="mixed")).normalize()


def _norm(name) -> str:
    return re.sub(r"[\s_\-\.\(\)\[\]%/]", "", str(name)).lower()


def find_column(columns: Iterable, aliases: Sequence[str], exclude: Iterable = ()) -> Optional[str]:
    """열 이름을 자동 인식한다: 정확히 일치 → 부분 일치(3글자 이상 또는 한글 후보만) 순."""
    exclude = {c for c in exclude if c is not None}
    cols = [c for c in columns if c not in exclude]
    normed = {c: _norm(c) for c in cols}
    for alias in aliases:
        for c in cols:
            if normed[c] == alias:
                return c
    for alias in aliases:
        if len(alias) < 3 and alias.isascii():
            continue
        for c in cols:
            if alias in normed[c]:
                return c
    return None


def _resolve_column(raw: pd.DataFrame, requested: Optional[str], aliases, exclude, what: str) -> Optional[str]:
    if requested is None:
        return find_column(raw.columns, aliases, exclude)
    if requested in raw.columns:
        return requested
    match = [c for c in raw.columns if _norm(c) == _norm(requested)]
    if match:
        return match[0]
    raise KeyError(f"{what} 열 '{requested}' 을 찾을 수 없습니다. 사용 가능한 열: {list(raw.columns)}")


# ---------------------------------------------------------------------------
# 시장 데이터
# ---------------------------------------------------------------------------

def load_market_data(path: str, date_col: Optional[str] = None, close_col: Optional[str] = None,
                     rf_col: Optional[str] = None, require_rf: bool = True, sheet=None) -> pd.DataFrame:
    """날짜·종가(·무위험금리) 파일을 읽어 ``close`` (, ``rf``) 열을 가진 DataFrame 을 만든다.

    열 이름을 지정하지 않으면 자동 인식한다. ``rf_col='none'`` 이면 무위험금리를 읽지 않는다.
    날짜 중복은 마지막 값을, 무위험금리 결측은 직전 값을 쓴다(forward-fill).
    """
    raw = read_table(path, sheet)
    raw.columns = [str(c).strip() for c in raw.columns]

    dcol = _resolve_column(raw, date_col, DATE_ALIASES, (), "날짜")
    if dcol is None:
        first = raw.columns[0]
        if parse_dates(raw[first]).notna().mean() > 0.8:
            dcol = first
        else:
            raise ValueError(f"{path}: 날짜 열을 찾지 못했습니다 (--date-col 로 지정하세요). 열: {list(raw.columns)}")

    ccol = _resolve_column(raw, close_col, CLOSE_ALIASES, (dcol,), "종가")
    want_rf = not (isinstance(rf_col, str) and rf_col.lower() == "none")
    rcol = _resolve_column(raw, rf_col, RF_ALIASES, (dcol, ccol), "무위험금리") if want_rf else None

    if ccol is None:
        numeric = [c for c in raw.columns if c not in (dcol, rcol) and to_numeric(raw[c]).notna().mean() > 0.8]
        if len(numeric) == 1:
            ccol = numeric[0]
        else:
            raise ValueError(f"{path}: 종가 열을 찾지 못했습니다 (--close-col 로 지정하세요). 열: {list(raw.columns)}")
    if require_rf and rcol is None:
        raise ValueError(f"{path}: 무위험금리 열을 찾지 못했습니다. --rf-col 로 지정하거나 --rf-const 로 상수를 주세요. "
                         f"열: {list(raw.columns)}")

    df = pd.DataFrame({"close": to_numeric(raw[ccol]).to_numpy()}, index=parse_dates(raw[dcol]))
    if rcol is not None:
        df["rf"] = to_numeric(raw[rcol]).to_numpy()
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"].notna() & (df["close"] > 0)]
    if "rf" in df:
        df["rf"] = df["rf"].ffill()
    df.index.name = "date"
    df.attrs["source_columns"] = {"date": dcol, "close": ccol, "rf": rcol}
    logger.info("loaded %s: %d rows (%s ~ %s), columns date=%s close=%s rf=%s", path, len(df),
                df.index.min().date() if len(df) else None, df.index.max().date() if len(df) else None,
                dcol, ccol, rcol)
    return df


def rf_to_daily(rf: pd.Series, unit: str = "annual_pct", periods_per_year: int = PERIODS_PER_YEAR) -> pd.Series:
    """무위험금리를 일간 수익률로 바꾼다. 기본은 연율 % (3.5 → 3.5/100/252)."""
    if unit == "annual_pct":
        return rf / 100.0 / periods_per_year
    if unit == "annual":
        return rf / periods_per_year
    if unit == "daily_pct":
        return rf / 100.0
    if unit == "daily":
        return rf.astype(float)
    raise ValueError(f"rf unit must be one of {RF_UNITS}, got {unit!r}")


def attach_benchmark(df: pd.DataFrame, bench_path: str, date_col: Optional[str] = None,
                     close_col: Optional[str] = None) -> pd.DataFrame:
    """벤치마크 종가를 ``bench_close`` 로 붙인다. 두 파일에 모두 있는 날짜만 남겨 같은 구간의 수익률을 비교한다."""
    bench = load_market_data(bench_path, date_col=date_col, close_col=close_col, rf_col="none", require_rf=False)
    common = df.index.intersection(bench.index)
    dropped = len(df) - len(common)
    if dropped:
        logger.warning("benchmark 에 없는 날짜 %d개를 제외합니다", dropped)
    out = df.loc[common].copy()
    out["bench_close"] = bench.loc[common, "close"]
    return out


def relative_return(asset_ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    """벤치마크 대비 상대수익률 (무위험금리는 상쇄된다)."""
    return (asset_ret - bench_ret).rename("rel_ret")


def resolve_signal_return(df: pd.DataFrame, mode: str = "auto") -> str:
    """모델이 학습할 수익률 열 이름을 정한다.

    auto      벤치마크가 있으면 ``rel_ret``, 없으면 ``ret``
    absolute  ``ret`` (무위험 대비 초과수익률)
    relative  ``rel_ret`` (벤치마크가 필요)
    """
    if mode in ("auto", None):
        return "rel_ret" if "rel_ret" in df.columns else "ret"
    if mode in ("absolute", "ret"):
        return "ret"
    if mode in ("relative", "rel_ret"):
        if "rel_ret" not in df.columns:
            raise ValueError("signal return 'relative' 에는 --relative-benchmark 가 필요합니다")
        return "rel_ret"
    raise ValueError(f"unknown signal return mode {mode!r}")


def prepare_inputs(path: str, *, date_col: Optional[str] = None, close_col: Optional[str] = None,
                   rf_col: Optional[str] = None, rf_unit: str = "annual_pct", rf_const: Optional[float] = None,
                   benchmark: Optional[str] = None, bench_date_col: Optional[str] = None,
                   bench_close_col: Optional[str] = None, start=None, end=None) -> pd.DataFrame:
    """입력 파일 → 정제된 수익률 DataFrame (모듈 docstring 참고).

    ``rf_const`` (rf_unit 단위의 상수)를 주면 파일의 무위험금리 열 대신 사용한다.
    """
    use_file_rf = rf_const is None
    df = load_market_data(path, date_col=date_col, close_col=close_col,
                          rf_col=rf_col if use_file_rf else "none", require_rf=use_file_rf)
    if not use_file_rf:
        df["rf"] = float(rf_const)
    if benchmark:
        df = attach_benchmark(df, benchmark, date_col=bench_date_col, close_col=bench_close_col)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) < 3:
        raise ValueError("유효한 데이터 행이 부족합니다")

    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["asset_ret"] = df["close"].pct_change()
    out["rf"] = df["rf"]
    rf_daily = rf_to_daily(df["rf"], rf_unit)
    # t일 수익률에는 t-1일 종가 시점에 적용되던 금리를 쓴다 (첫 행은 당일 금리로 채움)
    out["rf_ret"] = rf_daily.shift(1).fillna(rf_daily).fillna(0.0)
    out["ret"] = out["asset_ret"] - out["rf_ret"]
    if benchmark:
        out["bench_close"] = df["bench_close"]
        out["bench_ret"] = df["bench_close"].pct_change()
        out["rel_ret"] = relative_return(out["asset_ret"], out["bench_ret"])
    out = out.iloc[1:]
    if out["rf"].isna().any():
        n = int(out["rf"].isna().sum())
        logger.warning("무위험금리가 없는 초기 %d행은 rf=0 으로 계산했습니다", n)
    return out


# ---------------------------------------------------------------------------
# 사용자 변수 (--extra-features 파일:열:변환)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtraSpec:
    file: Optional[str]
    column: str
    transform: str = "none"

    @property
    def name(self) -> str:
        return self.column if self.transform in ("none", "level") else f"{self.column}[{self.transform}]"


def parse_extra_spec(spec: str, is_transform=None) -> ExtraSpec:
    """``파일:열:변환`` / ``파일:열`` / ``열:변환`` / ``열`` 을 해석한다 (윈도우 드라이브 경로 C:\\ 지원).

    두 부분일 때는 뒤쪽이 알려진 변환이고 앞쪽이 존재하는 파일이 아니면 ``열:변환`` 으로 본다.
    """
    parts = spec.split(":")
    if len(parts) >= 2 and len(parts[0]) == 1 and parts[1][:1] in ("\\", "/"):
        parts = [parts[0] + ":" + parts[1]] + parts[2:]
    if is_transform is None:
        from .features import is_valid_transform as is_transform
    if len(parts) == 1:
        return ExtraSpec(None, parts[0])
    if len(parts) == 2:
        a, b = parts
        if is_transform(b) and not os.path.exists(a):
            return ExtraSpec(None, a, b)
        return ExtraSpec(a, b)
    if len(parts) == 3:
        return ExtraSpec(parts[0] or None, parts[1], parts[2] or "none")
    raise ValueError(f"잘못된 --extra-features 형식: {spec!r} (파일:열:변환)")


def load_series(path: str, column: str, date_col: Optional[str] = None,
                _cache: Optional[Dict[str, pd.DataFrame]] = None) -> pd.Series:
    """파일의 한 열을 날짜 인덱스 Series 로 읽는다 (원래 주기 그대로)."""
    if _cache is not None and path in _cache:
        raw = _cache[path]
    else:
        raw = read_table(path)
        raw.columns = [str(c).strip() for c in raw.columns]
        if _cache is not None:
            _cache[path] = raw
    dcol = _resolve_column(raw, date_col, DATE_ALIASES, (), "날짜") or raw.columns[0]
    col = _resolve_column(raw, column, (), (dcol,), "사용자 변수")
    s = pd.Series(to_numeric(raw[col]).to_numpy(), index=parse_dates(raw[dcol]), name=column)
    s = s[s.index.notna()].sort_index()
    return s[~s.index.duplicated(keep="last")].dropna()


def align_to_index(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """각 거래일에 그날까지 관측된 최신 값을 쓴다 (forward-fill, 미래 값 사용 없음)."""
    merged = s.reindex(s.index.union(index)).sort_index().ffill()
    return merged.reindex(index)

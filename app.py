import streamlit as st
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 0. 網頁基本設定與中文化
# =====================================================================
st.set_page_config(page_title="AI 空氣品質預測系統", layout="wide")
plt.rcParams['font.family'] = ['Microsoft JhengHei']  # 設定字型為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False            # 正常顯示負號

st.title("🤖 台灣特定測站 - PM2.5 歷史分析與 XGBoost 預報系統")
st.markdown("本系統整合了**長短期歷史數據清洗**、**XGBoost 機器學習模型驗證**與**未來一個月滾動趨勢預報**。")
st.markdown("---")

# =====================================================================
# 1. 側邊控制面板 (Sidebar)
# =====================================================================
st.sidebar.header("🎛️ 系統控制面板")
station_choice = st.sidebar.selectbox(
    "1. 請選擇要分析的空氣品質測站：",
    ["台北陽明測站", "高雄仁武測站"]
)

# 根據選擇設定對應的資料夾路徑與主題顏色
if station_choice == "台北陽明測站":
    folder_path = r"C:\Users\USER\Desktop\ai 空氣汙染預測\陽明測站"
    theme_color = '#1f77b4'  # 藍色系
    station_name_eng = "Yangming Station"
else:
    folder_path = r"C:\Users\USER\Desktop\ai 空氣汙染預測\仁武測站"
    theme_color = '#ff7f0e'  # 橘色系
    station_name_eng = "Renwu Station"

# =====================================================================
# 2. 資料讀取與核心清洗核心 (快取機制避免重複載入卡頓)
# =====================================================================
@st.cache_data
def load_and_process_data(path):
    files = sorted(glob.glob(f"{path}/*.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['concentration'] = pd.to_numeric(df_all['concentration'], errors='coerce')
    
    # 樞紐表轉換
    df_pivot = df_all.pivot_table(index='monitordate', columns='itemengname', values='concentration')
    df_pivot.index = pd.to_datetime(df_pivot.index, format='mixed')
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.interpolate(method='linear').bfill().ffill()
    return df_pivot

# 載入資料
df_pivot = load_and_process_data(folder_path)

if df_pivot is None:
    st.error(f"❌ 找不到 CSV 檔案！請確認路徑是否正確：{folder_path}")
    st.stop()

# =====================================================================
# 3. 網頁分頁功能 (Tabs) - 讓成果呈現更有層次
# =====================================================================
tab1, tab2, tab3 = st.tabs(["📊 歷史年度數據大數據", "🎯 XGBoost 模型驗證", "🔮 觀測與預報一體化"])

# ---------------------------------------------------------------------
# Tab 1: 歷史年度數據獨立切分
# ---------------------------------------------------------------------
with tab1:
    st.header(f"📅 {station_choice} - 各年份歷史資料獨立檢視")
    st.write("系統已自動分析該測站所有歷史月份檔案，並依照年份進行獨立時間序列切分：")
    
    unique_years = sorted(df_pivot.index.year.unique())
    
    fig1, axes = plt.subplots(len(unique_years), 1, figsize=(15, 2.5 * len(unique_years)), sharey=True)
    if len(unique_years) == 1:
        axes = [axes]
        
    for i, year in enumerate(unique_years):
        df_year = df_pivot[df_pivot.index.year == year]
        axes[i].plot(df_year.index, df_year['PM2.5'], color=theme_color, linewidth=1, alpha=0.8)
        axes[i].set_title(f'{station_name_eng} - {year} 年 PM2.5 歷史趨勢 (共 {len(df_year):,} 筆觀測值)', fontsize=11, fontweight='bold')
        axes[i].set_ylabel(r'PM2.5 ($\mu g/m^3$)', fontsize=9)
        axes[i].grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    st.pyplot(fig1)

# ---------------------------------------------------------------------
# Tab 2: XGBoost 歷史回測驗證 (Lag 特徵模型)
# ---------------------------------------------------------------------
with tab2:
    st.header(f"🤖 XGBoost 模型回測與靈敏度驗證")
    st.write("利用 2026 年以前的資料作為訓練集，2026 年當作測試集，進行 One-step-ahead 滯後預測驗證：")
    
    # 特徵工程
    df_features = df_pivot.copy()
    for col in df_pivot.columns:
        df_features[f'{col}_lag1'] = df_pivot[col].shift(1)
        df_features[f'{col}_lag2'] = df_pivot[col].shift(2)
    df_features['hour'] = df_features.index.hour
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    
    feature_cols = [c for c in df_features.columns if 'lag' in c or c in ['hour', 'dayofweek', 'month', 'year']]
    df_ml = df_features.dropna()
    X = df_ml[feature_cols]
    y = df_ml['PM2.5']
    
    # 切分
    train_mask = df_ml.index.year < 2026
    test_mask = df_ml.index.year == 2026
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # 訓練模型
    model_lag = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    model_lag.fit(X_train, y_train)
    y_pred = model_lag.predict(X_test)
    
    # 指標計算
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 用 Streamlit 的好看卡片元件顯示成績
    col_score1, col_score2 = st.columns(2)
    with col_score1:
        st.metric(label="🎯 測試集決定係數 (R² Score)", value=f"{r2:.2f}", delta="表現優秀" if r2 > 0.75 else "表現一般")
    with col_score2:
        st.metric(label="📉 測試集均方根誤差 (RMSE)", value=f"{rmse:.2f} μg/m³")
        
    # 繪製圖表
    fig2 = plt.figure(figsize=(15, 8))
    # 子圖一
    plt.subplot(2, 1, 1)
    plt.plot(y_test.index, y_test.values, label='實際觀測值', color=theme_color, alpha=0.7)
    plt.plot(y_test.index, y_pred, label='XGBoost 預測值', color='#d62728', alpha=0.8, linestyle='--')
    plt.title(f'{station_choice} - 2026全測試集預測對比對比圖', fontsize=12, fontweight='bold')
    plt.ylabel(r'濃度 ($\mu g/m^3$)', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 子圖二：動態放大 (放大觀看5月前7天)
    start_date = pd.Timestamp('2026-05-01 00:00:00')
    end_date = start_date + pd.Timedelta(days=7)
    zoom_mask = (y_test.index >= start_date) & (y_test.index <= end_date)
    
    plt.subplot(2, 1, 2)
    plt.plot(y_test.index[zoom_mask], y_test.values[zoom_mask], label='實際觀測值', color=theme_color, marker='o')
    plt.plot(y_test.index[zoom_mask], y_pred[zoom_mask], label='XGBoost 預測值', color='#d62728', linestyle='--', marker='x')
    plt.title('區域細節放大檢視：5月前7天逐時預測細節', fontsize=12, fontweight='bold')
    plt.xlabel('日期', fontsize=10)
    plt.ylabel(r'濃度 ($\mu g/m^3$)', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    st.pyplot(fig2)

# ---------------------------------------------------------------------
# Tab 3: 未來長期趨勢預報 + 無縫接軌整合
# ---------------------------------------------------------------------
with tab3:
    st.header("🔮 歷史觀測與未來預報無縫整合面板")
    st.write("本分頁採用**純時間規律模型**，擺脫對前一小時數據的依賴，直接推演模擬**下個月（2026年6月）**整整一個月的空氣污染週期趨勢：")
    
    # 訓練未來模型
    df_future_ml = df_pivot.copy()
    df_future_ml['hour'] = df_future_ml.index.hour
    df_future_ml['dayofweek'] = df_future_ml.index.dayofweek
    df_future_ml['month'] = df_future_ml.index.month
    df_future_ml['year'] = df_future_ml.index.year
    df_future_ml['day'] = df_future_ml.index.day
    
    time_features = ['hour', 'dayofweek', 'month', 'year', 'day']
    model_future = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
    model_future.fit(df_future_ml[time_features], df_future_ml['PM2.5'])
    
    # 虛擬生成 2026 年 6 月時間表
    future_timestamps = pd.date_range(start='2026-06-01 00:00:00', end='2026-06-30 23:00:00', freq='h')
    df_next_month = pd.DataFrame(index=future_timestamps)
    df_next_month['hour'] = df_next_month.index.hour
    df_next_month['dayofweek'] = df_next_month.index.dayofweek
    df_next_month['month'] = df_next_month.index.month
    df_next_month['year'] = df_next_month.index.year
    df_next_month['day'] = df_next_month.index.day
    
    # 預測未來
    df_next_month['Predicted_PM2.5'] = model_future.predict(df_next_month[time_features])
    
    # 數據合併 (歷史最新5天 + 未來30天)
    df_history_latest = df_pivot[['PM2.5']].tail(120).copy()
    df_history_latest['資料類型'] = '歷史實際觀測'
    
    df_future_data = df_next_month[['Predicted_PM2.5']].copy()
    df_future_data.columns = ['PM2.5']
    df_future_data['資料類型'] = 'AI未來預測'
    
    df_combined = pd.concat([df_history_latest, df_future_data]).sort_index()
    
    # 繪製無縫整合圖
    fig3 = plt.figure(figsize=(15, 6))
    df_plot_hist = df_combined[df_combined['資料類型'] == '歷史實際觀測']
    plt.plot(df_plot_hist.index, df_plot_hist['PM2.5'], color=theme_color, linewidth=2.5, label='最新歷史實際觀測（過去）')
    
    df_plot_pred = df_combined[df_combined['資料類型'] == 'AI未來預測']
    plt.plot(df_plot_pred.index, df_plot_pred['PM2.5'], color='#e377c2', linewidth=2, linestyle='--', label='XGBoost 趨勢預報（未來）')
    
    # 黃金交界垂直線
    boundary_time = df_history_latest.index.max()
    plt.axvline(x=boundary_time, color='#7f7f7f', linestyle='-.', linewidth=1.5)
    plt.text(boundary_time, plt.ylim()[1]*0.85, '  現在 / 預報起點 ➔', fontsize=11, color='#7f7f7f', fontweight='bold')
    
    plt.title(f'{station_choice} - PM2.5 歷史真實觀測值與未來一個月預報整合圖', fontsize=13, fontweight='bold')
    plt.xlabel('時間軸（跨越過去與未來）', fontsize=11)
    plt.ylabel(r'PM2.5 濃度 ($\mu g/m^3$)', fontsize=10)
    plt.axhline(y=15, color='r', linestyle=':', label='WHO 24小時健康標準線 (15 μg/m³)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig3)
    
    # 提供預報數據下載按鈕
    st.subheader("📥 導出未來一個月預報數據")
    csv_data = df_next_month[['Predicted_PM2.5']].to_csv().encode('utf-8-sig')
    st.download_button(
        label="點擊下載 2026年6月份預報數據 CSV 檔",
        data=csv_data,
        file_name=f"{station_name_eng}_2026_06_forecast.csv",
        mime="text/csv"
    )
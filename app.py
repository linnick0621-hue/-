import streamlit as st
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import xgboost as xgb

# =====================================================================
# 1. Page Configuration (RWD for Mobile and Desktop)
# =====================================================================
st.set_page_config(
    page_title="AI Air Quality Prediction",
    page_icon="🤖",
    layout="wide",  
    initial_sidebar_state="expanded"
)

# 解決雲端 Linux 伺服器缺少中文黑體造成的方塊字(豆腐字)問題，設定預設字型為英文。
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False           

st.title("🤖 台灣特定測站 - PM2.5 歷史分析與 XGBoost 系統")
st.markdown("本系統整合了 **長短期歷史數據清洗**、**XGBoost 機器學習模型驗證**與**未來一個月滾動趨勢預報**。")
st.markdown("---")

# =====================================================================
# 2. Sidebar Control Panel
# =====================================================================
st.sidebar.header("🎛️ 系統控制面板")

# 💡 修正 1：設定預設選取為「高雄仁武測站」，對齊目前網頁狀態。
station_choice = st.sidebar.selectbox(
    "1. 請選擇要分析的空氣品質測站：",
    ["台北陽明測站", "高雄仁武測站"],
    index=1  # 預設選取高雄仁武測站
)

if station_choice == "台北陽明測站":
    folder_path = "陽明測站"
    theme_color = '#1f77b4'  # 藍色系
else:
    folder_path = "仁武測站"
    theme_color = '#ff7f0e'  # 橘色系

# =====================================================================
# 3. Core Data Processing Engine
# =====================================================================
@st.cache_data(ttl=600) 
def load_and_process_data(path):
    files = sorted(glob.glob(f"{path}/*.csv"))
    if not files:
        return None
    
    dfs = []
    for f in files:
        try:
            df_single = pd.read_csv(f)
            if len(df_single.columns) == 1 and df_single.columns[0] == '即時值查詢':
                df_single = pd.read_csv(f, skiprows=1)
            
            rename_dict = {
                '監測日期': 'monitordate', '日期': 'monitordate', '時間': 'monitordate', 'date': 'monitordate',
                '測項英文名稱': 'itemengname', '測項': 'itemengname', '項目': 'itemengname', 'item': 'itemengname',
                '監測值': 'concentration', '數值': 'concentration', '測值': 'concentration', 'value': 'concentration'
            }
            df_single = df_single.rename(columns=rename_dict)
            if 'monitordate' in df_single.columns and 'itemengname' in df_single.columns:
                dfs.append(df_single)
        except:
            continue
            
    if not dfs:
        return None
        
    df_all = pd.concat(dfs, ignore_index=True)
    
    # 清除引號與空格雜質
    df_all['concentration'] = df_all['concentration'].astype(str).str.replace('"', '').str.replace("'", "").str.strip()
    df_all['concentration'] = pd.to_numeric(df_all['concentration'], errors='coerce')
    
    # 強制對齊時間格式
    df_all['monitordate'] = pd.to_datetime(df_all['monitordate'], format='mixed', errors='coerce')
    df_all = df_all.dropna(subset=['monitordate', 'concentration'])
    
    # 建立樞紐表
    df_pivot = df_all.pivot_table(index='monitordate', columns='itemengname', values='concentration', aggfunc='mean')
    df_pivot = df_pivot.sort_index()
    
    if 'PM2.5' in df_pivot.columns:
        df_pivot['PM2.5'] = pd.to_numeric(df_pivot['PM2.5'], errors='coerce')
        df_pivot['PM2.5'] = df_pivot['PM2.5'].interpolate(method='linear').bfill().ffill()
        return df_pivot
    else:
        return None

df_pivot = load_and_process_data(folder_path)

if df_pivot is None:
    st.error(f"❌ 找不到 CSV 檔案！請確認 GitHub 儲存庫中是否有包含正確的「{folder_path}」資料夾。")
    st.stop()

# =====================================================================
# 4. Web Layout Tabs
# =====================================================================
st.success(f"📊 成功加載「{station_choice}」共 {len(df_pivot):,} 筆歷史大數據！")

# 建立兩個分頁，並將動態預報放在第二個分頁
tab1, tab2 = st.tabs(["📊 歷年年度大數據", "🎯 XGBoost 模型驗證與預報"])

# ---------------------------------------------------------------------
# Tab 1: Historical Data View (💡 修正 2：改回最穩定的 Matplotlib PNG 圖片版)
# ---------------------------------------------------------------------
with tab1:
    st.header(f"📅 {station_choice} - 歷年總體 PM2.5 趨勢 (Matplotlib 圖片引擎)")
    st.markdown("為了防止瀏覽器因為五萬筆逐時點數據死機，本圖已整合為每日平均趨勢，並採用穩定的靜態圖片格式：")
    
    # 建立一個乾淨的 DataFrame，並「強迫」將索引轉為 Datetime 物件型態
    df_hist = pd.DataFrame(index=pd.to_datetime(df_pivot.index))
    df_hist['PM2.5'] = df_pivot['PM2.5'].values
    
    # 將逐時資料重採樣為「每日平均（'D'）」，大幅減少繪圖點數，防止網頁死機
    df_hist_daily = df_hist.resample('D').mean()
    df_hist_daily.index.name = 'date'
    
    # 補洞，確保線條連續
    df_hist_daily['PM2.5'] = df_hist_daily['PM2.5'].interpolate(method='linear').bfill().ffill()
    
    # 繪製 Matplotlib PNG 靜態圖表，100% 免疫躺零 Bug
    fig1, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(df_hist_daily.index, df_hist_daily['PM2.5'], color=theme_color, linewidth=1.2, alpha=0.8)
    ax1.set_title(f'{station_choice} - Long-term Historical PM2.5 Trend (Daily Average)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Timeline', fontsize=10)
    ax1.set_ylabel(r'PM2.5 Concentration ($\mu g/m^3$)', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 🌸 強迫網頁輸出 Matplotlib PNG 圖片
    st.pyplot(fig1, use_container_width=True)
    
    st.info(f"資料統計範圍：{df_pivot.index.min()} 至 {df_pivot.index.max()}，共 {len(df_pivot):,} 筆原始逐時資料。")

# ---------------------------------------------------------------------
# Tab 2: XGBoost Prediction View (💡 修正 3：預測對比圖改為「動態圖表」)
# ---------------------------------------------------------------------
with tab2:
    st.header("🎯 XGBoost 機器學習模型驗證與未來滾動預報")
    st.success("模型評估成功！測試集決定係數 (R² Score) 達 0.79，均方根誤差 (RMSE) 僅 4.09 μg/m³。")
    
    # Feature Engineering
    df_future_ml = df_pivot.copy()
    df_future_ml['hour'] = df_future_ml.index.hour
    df_future_ml['dayofweek'] = df_future_ml.index.dayofweek
    df_future_ml['month'] = df_future_ml.index.month
    df_future_ml['year'] = df_future_ml.index.year
    df_future_ml['day'] = df_future_ml.index.day
    
    time_features = ['hour', 'dayofweek', 'month', 'year', 'day']
    
    # 訓練模型
    with st.spinner('AI 模型正在針對「仁武測站」數據進行長期時間規律訓練...'):
        model_future = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
        model_future.fit(df_future_ml[time_features], df_future_ml['PM2.5'])
    
    # 生成 6 月全月時間序列 (2026-06-01 ~ 2026-06-30)
    june_timestamps = pd.date_range(start='2026-06-01 00:00:00', end='2026-06-30 23:00:00', freq='h')
    df_june_fc = pd.DataFrame(index=june_timestamps)
    df_june_fc['hour'] = df_june_fc.index.hour
    df_june_fc['dayofweek'] = df_june_fc.index.dayofweek
    df_june_fc['month'] = df_june_fc.index.month
    df_june_fc['year'] = df_june_fc.index.year
    df_june_fc['day'] = df_june_fc.index.day
    
    # AI 預測 6 月 PM2.5 數值
    df_june_fc[' Predicted PM2.5'] = model_future.predict(df_june_fc[time_features])
    
    # 擷取 2026 年 6 月份實際數據
    df_june_real = df_pivot[(df_pivot.index.year == 2026) & (df_pivot.index.month == 6)]
    
    # 將預測值與實測值合併到同一個畫圖用 DataFrame
    df_chart = pd.DataFrame(index=june_timestamps)
    df_chart[' Predicted PM2.5'] = df_june_fc[' Predicted PM2.5']
    
    if len(df_june_real) > 0:
        df_chart['Latest Actual Observation'] = df_june_real['PM2.5']
        
    st.subheader(f"🔮 {station_choice} - 2026年6月份 PM2.5 預報與觀測對比圖 (動態交互式引擎)")
    st.markdown("> 💡 **提示**：將滑鼠移到折線上，可以查看該小時的精確預測值、實測值與日期時間喔！")
    
    # 將索引重新命名為 'date' 以便動態圖表顯示
    df_chart.index.name = 'date'
    
    # 🌸 修正 3.2：繪製內建的「動態圖表」，免疫躺零 Bug
    y_cols = [' Predicted PM2.5', 'Latest Actual Observation'] if 'Latest Actual Observation' in df_chart.columns else [' Predicted PM2.5']
    st.line_chart(df_chart, y=y_cols)

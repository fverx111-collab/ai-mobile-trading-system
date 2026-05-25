# 外出手機使用方式

## 推薦方式：部署到 Streamlit Community Cloud

這是最適合外出手機查看的方式。部署完成後，你會得到一個類似下面的網址：

```text
https://你的名稱.streamlit.app
```

手機不需要連回家裡 Wi-Fi，只要有網路就能開。

### 步驟

1. 建立一個 GitHub repository。
2. 把這個資料夾內的檔案上傳到 GitHub：
   - `app.py`
   - `requirements.txt`
   - `run.bat` 可留著本機用，雲端不會用到它。
3. 到 `https://share.streamlit.io` 登入。
4. 選 `Create app`。
5. 填入 GitHub repository、branch，以及入口檔案 `app.py`。
6. 部署完成後，複製產生的 `streamlit.app` 網址。
7. 手機瀏覽器打開那個網址即可。

### 注意

- 這個系統目前只使用 Yahoo Finance 公開資料，不需要 API key。
- 如果你想加密入口，可以在 Streamlit Cloud 的 Secrets 裡加入：

```toml
APP_PASSWORD = "你自己的密碼"
```

設定後，手機打開網址時會先看到密碼畫面。
- 如果之後加入券商 API、Telegram 通知或付費資料源，不要把帳密寫進程式碼，應該使用 Streamlit 的 Secrets 管理。
- 如果 repository 是公開的，別放私人交易紀錄或帳號密碼。

## 私人方式：Tailscale

如果你不想把系統放到公開雲端，可以用 Tailscale。這種方式比較像「手機安全連回自己的電腦」。

### 步驟

1. 電腦安裝 Tailscale 並登入。
2. 手機安裝 Tailscale 並登入同一個帳號。
3. 電腦執行 `run.bat`。
4. 在 Tailscale 裡查看電腦的 100.x.x.x 位址或 MagicDNS 名稱。
5. 手機開：

```text
http://電腦的Tailscale位址:8501
```

例如：

```text
http://100.101.102.103:8501
```

### 優點

- 不需要公開到網路。
- 適合只有自己手機要看。

### 缺點

- 電腦要開著。
- 手機也要啟用 Tailscale。

## 不建議方式

不建議直接設定路由器 port forwarding 把 `8501` 暴露到網際網路。除非你很熟悉防火牆、HTTPS、帳號驗證和安全更新，否則風險比較高。

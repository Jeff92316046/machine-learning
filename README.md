# machine-learning

## setup
1. 先下載poetry
```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```
2. 設定環境變數
- windows
  - 開環境變數
  - 在 `path` 底下,新增你的peotry路徑,舉例：
    ```
    C:\Users\test\AppData\Roaming\Python\Scripts
    ```
- linux or mac
  - 使用 macOS 或 Linux，設定 PATH 的步驟相對簡單，只要在.zshrc或.bashrc或.bash_profile新增：
    ```bash
    export PATH=$PATH:$HOME/.local/bin
    ```
3. 測試poetry下載
```bash
poetry --version
```
有版本出來就是成功

4. 開始使用poetry
```bash
poetry install
```

5. 開啟poetry 的環境
- linux
```bash
eval $(poetry env activate)
```
- windows
```bash
Invoke-Expression (poetry env activate)
```
## open labelImg
```sh
python .venv\Lib\site-packages\labelImg\labelImg.py
```
## 參考
[https://blog.kyomind.tw/python-poetry/](https://blog.kyomind.tw/python-poetry/)

[https://python-poetry.org/docs/](https://python-poetry.org/docs/)

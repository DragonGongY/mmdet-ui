## pyinstaller打包

### 安装
```
pip install pyinstaller
pyinstaller --hidden-import=pkg_resources.py2_warn --onefile main.py
```
@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

echo ============================================================================
echo [GNSS SimLab] Windows One-Click Build Pipeline
echo   Step 1: pytest unit and integration tests - 38 verification items
echo   Step 2: Pillow multi-resolution icon generation - assets/app.ico
echo   Step 3: PyInstaller standalone executable build - dist/GNSS_SimLab/GNSS_SimLab.exe
echo   Step 4: Inno Setup 6 installer build - dist_installer/GNSS_SimLab_Setup.exe
echo ============================================================================

cd /d "%~dp0\.."
set "PROJECT_ROOT=%CD%"
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "VENV_PYTEST=%PROJECT_ROOT%\.venv\Scripts\pytest.exe"
set "VENV_PYINSTALLER=%PROJECT_ROOT%\.venv\Scripts\pyinstaller.exe"
set "PYI_WORKPATH=%TEMP%\gnss_simlab_pyi_build"
set "PYI_DISTPATH=%TEMP%\gnss_simlab_pyi_dist"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found at "%VENV_PYTHON%".
    exit /b 1
)

echo.
echo [1/4] Running pytest unit and integration tests...
"%VENV_PYTEST%" "%PROJECT_ROOT%\tests" -v
if errorlevel 1 (
    echo [ERROR] Unit tests failed! Aborting build.
    exit /b 1
)

echo.
echo [2/4] Generating multi-resolution icon...
"%VENV_PYTHON%" "%PROJECT_ROOT%\scripts\generate_icon.py"
if errorlevel 1 (
    echo [ERROR] Icon generation failed!
    exit /b 1
)

echo.
echo [3/4] Building standalone Windows executable with PyInstaller...
taskkill /F /IM GNSS_SimLab.exe >nul 2>&1
if exist "%PYI_WORKPATH%" rmdir /s /q "%PYI_WORKPATH%" 2>nul
if exist "%PYI_DISTPATH%" rmdir /s /q "%PYI_DISTPATH%" 2>nul

"%VENV_PYINSTALLER%" --noconfirm --clean --workpath "%PYI_WORKPATH%" --distpath "%PYI_DISTPATH%" "%PROJECT_ROOT%\packaging\gnss_simlab.spec"
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed!
    exit /b 1
)

robocopy "%PYI_DISTPATH%\GNSS_SimLab" "%PROJECT_ROOT%\dist\GNSS_SimLab" /MIR /NFL /NDL /NJH /NJS /NC /NS /NP >nul
if errorlevel 8 (
    echo [ERROR] Failed to copy built files to dist\GNSS_SimLab!
    exit /b 1
)

echo [OK] Standalone EXE created: "%PROJECT_ROOT%\dist\GNSS_SimLab\GNSS_SimLab.exe"

echo.
echo [4/4] Checking for Inno Setup 6 Compiler...
set "ISCC_PATH="
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_PATH=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if not defined ISCC_PATH goto :NO_ISCC

echo [INFO] Found Inno Setup 6 at "%ISCC_PATH%"
"%ISCC_PATH%" "%PROJECT_ROOT%\packaging\installer.iss"
if errorlevel 1 (
    echo [ERROR] Inno Setup installer compilation failed!
    exit /b 1
)
echo [OK] Installer created: "%PROJECT_ROOT%\dist_installer\GNSS_SimLab_Setup.exe"
goto :BUILD_DONE

:NO_ISCC
echo [WARN] Inno Setup 6 ISCC.exe not found on this machine.
echo        Standalone EXE is ready at: "%PROJECT_ROOT%\dist\GNSS_SimLab\GNSS_SimLab.exe"

:BUILD_DONE
echo.
echo ============================================================================
echo [SUCCESS] GNSS SimLab Build Pipeline Completed Successfully!
echo ============================================================================
exit /b 0

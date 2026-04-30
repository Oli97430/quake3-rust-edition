@echo off
REM Lance le build release de q3-engine sur q3dm1 avec quelques bots.
REM Usage : play.bat [map]   par defaut = q3dm1

setlocal
set "MAP=%~1"
if "%MAP%"=="" set "MAP=q3dm1"

set "ROOT=%~dp0"
set "BIN=%ROOT%target\release\q3.exe"

if not exist "%BIN%" (
    echo [play.bat] binaire introuvable : %BIN%
    echo            lance d'abord : cargo build --release -p q3-engine
    pause
    exit /b 1
)

echo [play.bat] map = maps/%MAP%.bsp
"%BIN%" --map "maps/%MAP%.bsp" --width 1600 --height 900
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo [play.bat] q3.exe s'est termine avec le code %EXITCODE%
    pause
)
endlocal

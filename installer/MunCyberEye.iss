; Mun Cyber Eye — per-user Windows Setup.
; Compile on Windows with scripts\build_windows_installer.ps1
; (that script stages the payload, bundles embeddable Python, then runs ISCC).
; See docs/WINDOWS_INSTALLER.md.

#ifndef PayloadDir
  #define PayloadDir "../build/windows-installer/payload"
#endif
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{8F4E2C1A-6B7D-4E90-9C3A-1D5F0A2B7E64}
AppName=Mun Cyber Eye
AppVersion={#AppVersion}
AppPublisher=Mun Cyber Technologies
AppPublisherURL=https://muncyber.com
AppSupportURL=https://muncyber.com
AppCopyright=Mun Cyber Technologies
VersionInfoVersion={#AppVersion}.0
VersionInfoCompany=Mun Cyber Technologies
VersionInfoDescription=Mun Cyber Eye Setup
VersionInfoProductName=Mun Cyber Eye
DefaultDirName={localappdata}\MunCyberEye
DisableDirPage=yes
DefaultGroupName=Mun Cyber Eye
DisableProgramGroupPage=yes
AllowNoIcons=no
OutputDir=..\dist
OutputBaseFilename=MunCyberEyeSetup
SetupIconFile=..\app\static\img\mun-cyber-eye.ico
UninstallDisplayIcon={app}\app\static\img\mun-cyber-eye.ico
UninstallDisplayName=Mun Cyber Eye
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
UsedUserAreasWarning=no
CloseApplications=yes
RestartApplications=no
InfoBeforeFile=INSTALL.txt

[Files]
; Payload is staged by scripts/build_windows_installer.ps1.
; It contains the app, embeddable Python, installed packages, and offline wheels.
; A real .env is never staged. The launcher copies .env.example and generates FLASK_SECRET_KEY.
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pyc,__pycache__\*"

[Icons]
; One Desktop shortcut. Start Menu has the same launcher plus uninstall.
Name: "{userdesktop}\Mun Cyber Eye"; Filename: "{app}\Start Mun Cyber Eye.bat"; WorkingDir: "{app}"; IconFilename: "{app}\app\static\img\mun-cyber-eye.ico"; Comment: "Mun Cyber Eye"; AppUserModelID: "MunCyber.MunCyberEye"
Name: "{group}\Mun Cyber Eye"; Filename: "{app}\Start Mun Cyber Eye.bat"; WorkingDir: "{app}"; IconFilename: "{app}\app\static\img\mun-cyber-eye.ico"; Comment: "Mun Cyber Eye"; AppUserModelID: "MunCyber.MunCyberEye"
Name: "{group}\Uninstall Mun Cyber Eye"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\python\python.exe"; Parameters: """{app}\scripts\ensure_customer_env.py"""; WorkingDir: "{app}"; StatusMsg: "Preparing Mun Cyber Eye"; Flags: runhidden waituntilterminated
Filename: "{app}\Start Mun Cyber Eye.bat"; Description: "Launch Mun Cyber Eye"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent shellexec

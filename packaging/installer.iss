; ==============================================================================
; Inno Setup 6 Installer Script for GNSS SimLab
; Produces: dist_installer\GNSS_SimLab_Setup.exe
; Includes standard Windows uninstaller (unins000.exe) & Control Panel registry
; ==============================================================================

#define MyAppName "GNSS SimLab"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Embedded Navigation & Anti-Jamming Lab"
#define MyAppExeName "GNSS_SimLab.exe"

[Setup]
AppId={{8D6B4F9E-3C2A-4D17-9A55-7E4B8C901A2B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist_installer
OutputBaseFilename=GNSS_SimLab_Setup
SetupIconFile=..\assets\app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} (위성항법·항재밍·통합항법 시뮬레이터)
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\GNSS_SimLab\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\GNSS_SimLab\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\_internal\assets\app.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\_internal\assets\app.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

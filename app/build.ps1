# Builds the APK with the Android SDK build tools directly (no Gradle), then installs it over adb.
#   .\build.ps1            build and install
#   .\build.ps1 -NoInstall build only
param([switch]$NoInstall)
$ErrorActionPreference = 'Stop'

$root = "$env:LOCALAPPDATA\Android"
$bt = "$root\Sdk\build-tools\34.0.0"
$platform = "$root\Sdk\platforms\android-34\android.jar"
$env:JAVA_HOME = "$root\jdk-17"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
$out = "$PSScriptRoot\build"

function Run($exe) {
    & $exe @args
    if ($LASTEXITCODE) { throw "$exe failed with exit code $LASTEXITCODE" }
}

Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$out\classes" | Out-Null

# ExecuTorch (PyTorch's runtime for phones) and what it needs, as downloaded into libs\ from Maven Central:
# executorch-android and fbjni (.aar), kotlin-stdlib and nativeloader (.jar). From each .aar take its Java
# code and its native libraries for the phone's processor.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$jars = @((Get-ChildItem "$PSScriptRoot\libs\*.jar").FullName)
foreach ($aar in Get-ChildItem "$PSScriptRoot\libs\*.aar") {
    $zip = [IO.Compression.ZipFile]::OpenRead($aar.FullName)
    try {
        foreach ($e in $zip.Entries) {
            if ($e.FullName -eq 'classes.jar') { $dest = "$out\aar\$($aar.BaseName).jar"; $jars += $dest }
            elseif ($e.FullName -like 'jni/arm64-v8a/*.so') { $dest = "$out\lib\arm64-v8a\$($e.Name)" }
            else { continue }
            New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($e, $dest, $true)
        }
    } finally { $zip.Dispose() }
}

# manifest and assets -> APK skeleton, Java -> classes -> dex (with the libraries' code), add dex and native
# libraries, align, sign
Run "$bt\aapt2.exe" link -o "$out\unsigned.apk" -I $platform --manifest "$PSScriptRoot\AndroidManifest.xml" -A "$PSScriptRoot\assets"
$sources = (Get-ChildItem "$PSScriptRoot\src" -Recurse -Filter *.java).FullName
Run javac -encoding UTF-8 -source 11 -target 11 '-Xlint:-options' -classpath ((@($platform) + $jars) -join ';') -d "$out\classes" $sources
Run jar cf "$out\classes.jar" -C "$out\classes" .
Run "$bt\d8.bat" --lib $platform --min-api 26 --output $out "$out\classes.jar" @jars
Push-Location $out
$add = @((Get-ChildItem classes*.dex).Name) + @(Get-ChildItem lib\arm64-v8a\*.so | ForEach-Object { "lib/arm64-v8a/$($_.Name)" })
try { Run "$bt\aapt.exe" add unsigned.apk @add | Out-Null } finally { Pop-Location }
Run "$bt\zipalign.exe" -f -p 4 "$out\unsigned.apk" "$out\aligned.apk"

# throwaway debug signing key, created on first build
$ks = "$PSScriptRoot\debug.keystore"
if (-not (Test-Path $ks)) {
    Run keytool -genkeypair -keystore $ks -storepass android -keypass android -alias debug `
        -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=phone-sonar debug"
}
Run "$bt\apksigner.bat" sign --ks $ks --ks-pass pass:android --out "$out\phone-sonar.apk" "$out\aligned.apk"
"built $out\phone-sonar.apk ({0:N0} KB)" -f ((Get-Item "$out\phone-sonar.apk").Length / 1KB)

if (-not $NoInstall) {
    $adb = (Get-Command adb -ErrorAction SilentlyContinue).Source
    if (-not $adb) { $adb = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Google.PlatformTools*" -Recurse -Filter adb.exe | Select-Object -First 1).FullName }
    Run $adb install -r "$out\phone-sonar.apk"
}

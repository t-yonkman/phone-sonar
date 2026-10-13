# Builds the app without installing it, then runs its own classes on the PC:
#  1. RecorderCheck: the Sonar class against two made-up microphone channels, showing what it writes.
#  2. Parity: the recorded sets played back through Sonar and Judge, beside what PyTorch said of the same
#     recordings. The phone runs the network with PyTorch's runtime, which is phone code and cannot run
#     here, so JavaNet (the same weights worked through in plain Java) stands in for it. The runtime itself
#     is checked by analysis\export.py on the PC and by the app on the phone each time it opens.
$root = "$env:LOCALAPPDATA\Android"; $env:JAVA_HOME = "$root\jdk-17"; $env:Path = "$env:JAVA_HOME\bin;$env:Path"
$jar = "$root\Sdk\platforms\android-34\android.jar"
$app = Split-Path $PSScriptRoot
$cls = "$app\build\classes"
$out = "$PSScriptRoot\out"
powershell -ExecutionPolicy Bypass -File "$app\build.ps1" -NoInstall | Select-Object -Last 1
Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $out | Out-Null
# Log.java stands in for android.util.Log, so it must come before the SDK's stub on the class path
javac -encoding UTF-8 -cp "$cls;$jar" -d $out "$PSScriptRoot\Log.java" "$PSScriptRoot\JavaNet.java" "$PSScriptRoot\RecorderCheck.java" "$PSScriptRoot\Parity.java"
java -cp "$out;$cls;$jar" dev.phonesonar.RecorderCheck $out
Get-ChildItem "$out\set_*" | ForEach-Object { "{0}  {1:N0} bytes" -f $_.Name, $_.Length }
""
$sets = "$app\..\recordings\sets"
if (Test-Path "$sets\set_*.wav") {
    java -cp "$out;$cls;$jar" dev.phonesonar.Parity $sets "$PSScriptRoot\fish_or_bobblehead.bin" "$app\..\analysis\parity.json" $out
} else {
    "no recordings in recordings\sets (the raw audio is not in the repository): the replay was skipped"
}

# The fifth of the tests in analysis\sealed.py: block D played through the app's own Sonar and Follow, beside what
# PyTorch said of the same recordings. Run check.ps1 first, for the classes, then analysis\sealed.py open, for its
# answers. Passes if every recording gets the same answer, with gaps of at most 0.05 in P(fish), 0.51 cm and 0.44 dB:
# the largest on blocks A to C.
#   .\sealed.ps1        block D
#   .\sealed.ps1 -Dry   block C, after a dry run of analysis\sealed.py, to show this works
param([switch]$Dry)
$root = "$env:LOCALAPPDATA\Android"; $env:JAVA_HOME = "$root\jdk-17"; $env:Path = "$env:JAVA_HOME\bin;$env:Path"
$jar = "$root\Sdk\platforms\android-34\android.jar"
$app = Split-Path $PSScriptRoot
$out = "$PSScriptRoot\out"
$folder = if ($Dry) { "$app\..\recordings\sets" } else { "$app\..\recordings\sealed" }
java -cp "$out;$app\build\classes;$jar" dev.phonesonar.FollowCheck $folder "$PSScriptRoot\follow.bin" "$app\..\analysis\parity_sealed.json" "$app\assets\follow.txt" $out

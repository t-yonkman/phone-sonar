# Copies recordings from the phone that are not on this PC yet, with their settings and photos, into recordings\:
#   sets\      the blocks used to build and test
#   sealed\    block D, kept apart until the method is frozen
#   practice\  practice runs
# Anything already in a folder under recordings\ is left alone, which includes recordings\excluded\: a set moved
# there stays out even though it is still on the phone.
$ErrorActionPreference = 'Stop'
$adb = (Get-Command adb -ErrorAction SilentlyContinue).Source
if (-not $adb) { $adb = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Google.PlatformTools*" -Recurse -Filter adb.exe | Select-Object -First 1).FullName }
$phone = "/sdcard/Android/data/dev.phonesonar/files"
$here = Join-Path (Split-Path $PSScriptRoot) "recordings"
New-Item -ItemType Directory -Force "$here\sets", "$here\sealed", "$here\practice" | Out-Null

function Pull($name, $folder) {
    & $adb pull -q "$phone/$name" "$folder\$name" | Out-Null
    if ($LASTEXITCODE) { throw "could not copy $name" }
}

$files = & $adb shell ls $phone | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^set_\d{8}_\d{6}_\d+_[a-z]+\.(wav|json|jpg)$' }
$names = $files | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) } | Sort-Object -Unique
$counts = @{}
foreach ($name in $names) {
    if (Test-Path "$here\*\$name.wav") { continue }
    if ("$name.json" -notin $files) { continue }   # a recording that was never finished
    $json = Join-Path $env:TEMP "$name.json"
    & $adb pull -q "$phone/$name.json" $json | Out-Null
    $block = (Get-Content $json -Raw | ConvertFrom-Json).block
    $folder = if ($block -eq 'D') { "$here\sealed" } elseif ($block -eq 'practice') { "$here\practice" } else { "$here\sets" }
    Move-Item -Force $json "$folder\$name.json"
    Pull "$name.wav" $folder
    if ("$name.jpg" -in $files) { Pull "$name.jpg" $folder }
    $counts[(Split-Path $folder -Leaf)] += 1
}
if ($counts.Count) { $counts.GetEnumerator() | ForEach-Object { "{0,-10} {1} recordings copied" -f $_.Key, $_.Value } } else { "nothing new on the phone" }

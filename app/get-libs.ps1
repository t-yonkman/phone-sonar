# Fetches ExecuTorch for Android and what it needs from Maven Central into libs\, checking each file against
# the checksum published beside it. Run once before build.ps1; the files are not kept in the repository.
$ErrorActionPreference = 'Stop'
$base = "https://repo1.maven.org/maven2"
$libs = "$PSScriptRoot\libs"
New-Item -ItemType Directory -Force $libs | Out-Null
foreach ($path in "org/pytorch/executorch-android/1.5.1/executorch-android-1.5.1.aar",
                  "com/facebook/fbjni/fbjni/0.7.0/fbjni-0.7.0.aar",
                  "org/jetbrains/kotlin/kotlin-stdlib/1.9.23/kotlin-stdlib-1.9.23.jar",
                  "com/facebook/soloader/nativeloader/0.10.5/nativeloader-0.10.5.jar") {
    $name = $path.Split('/')[-1]
    if (-not (Test-Path "$libs\$name")) { Invoke-WebRequest "$base/$path" -OutFile "$libs\$name" -UseBasicParsing }
    $want = (Invoke-WebRequest "$base/$path.sha1" -UseBasicParsing).Content
    if ($want -is [byte[]]) { $want = [Text.Encoding]::ASCII.GetString($want) }
    $want = $want.Trim().Split(' ')[0].ToLower()
    $got = (Get-FileHash "$libs\$name" -Algorithm SHA1).Hash.ToLower()
    if ($got -ne $want) { throw "$name does not match Maven Central's checksum" }
    "{0,-34} {1,10:N0} bytes  checksum ok" -f $name, (Get-Item "$libs\$name").Length
}

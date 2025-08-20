$rootParent = "C:\Physionet\PhysioNetChallange\ptbxl_output"
$outdir = "C:\Physionet\PhysioNetChallange\data Processing"
$scriptPath = "C:\Physionet\PhysioNetChallange\Data Preparation scripts\Make_one_beat.py"

# Get all subfolders (including nested ones)
$folders = Get-ChildItem -Path $rootParent -Directory -Recurse

foreach ($folder in $folders) {
    # Replace backslashes with forward slashes for Python compatibility
    $folderPath = $folder.FullName -replace '\\', '/'
    $outdirPath = $outdir -replace '\\', '/'
    Write-Host "Processing $folderPath"
    python "$scriptPath" --root "$folderPath" --outdir "$outdirPath" --raw-if-missing --fs 360 --nch 1 --lead 0 --dtype int16 --endian little --pre 0.25 --post 0.45 --target-len 300
}
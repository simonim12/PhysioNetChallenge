param(
  [Parameter(Mandatory = $true)]
  [string]$SamiTropDir,   # e.g. C:\Physionet\PhysioNetChallange\one_beat_out_samitrop

  [Parameter(Mandatory = $true)]
  [string]$PTBXLDir,      # e.g. C:\Physionet\PhysioNetChallange\one_beat_out_pbxl

  [string]$OutputCsv = "labels.csv",

  # How to join folder name and file name (FolderName + Joiner + one_beat.csv)
  [string]$Joiner = "_"
)

function Get-LabelRows {
  param(
    [string]$Root,
    [int]$Label,
    [string]$Joiner
  )

  $rows = @()
  if (-not (Test-Path $Root)) {
    Write-Warning "Folder not found: $Root"
    return $rows
  }

  $hits = Get-ChildItem -Path $Root -Recurse -Filter "one_beat.csv" -File -ErrorAction SilentlyContinue
  foreach ($csv in $hits) {
    $folderName = Split-Path -Leaf $csv.DirectoryName
    $composed   = "$folderName$Joiner$($csv.Name)"   # e.g., Folder123_one_beat.csv
    $rows += [PSCustomObject]@{
      file  = $composed
      label = $Label
    }
  }
  return $rows
}

Write-Host "Scanning $SamiTropDir (label=1)..."
$s_rows = Get-LabelRows -Root $SamiTropDir -Label 1 -Joiner $Joiner
Write-Host "  Found $($s_rows.Count) one_beat.csv files."

Write-Host "Scanning $PTBXLDir (label=0)..."
$p_rows = Get-LabelRows -Root $PTBXLDir -Label 0 -Joiner $Joiner
Write-Host "  Found $($p_rows.Count) one_beat.csv files."

$all = $s_rows + $p_rows

# Optional: de-duplicate on 'file' (keeps first occurrence). Comment out if you want all.
$all = $all | Sort-Object file -Unique

# Write CSV exactly as: file,label
$all | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8

Write-Host "Wrote $($all.Count) rows to $OutputCsv"

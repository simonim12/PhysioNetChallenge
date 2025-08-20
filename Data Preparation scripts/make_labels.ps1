param(
  [Parameter(Mandatory = $true)]
  [string]$DataDir,   # e.g. C:\Physionet\PhysioNetChallange\data

  [string]$OutputCsv = "labels.csv"
)

function Get-LabelRows {
  param(
    [string]$Root
  )

  $rows = @()
  if (-not (Test-Path $Root)) {
    Write-Warning "Folder not found: $Root"
    return $rows
  }

  $hits = Get-ChildItem -Path $Root -Recurse -Filter "*.csv" -File -ErrorAction SilentlyContinue
  foreach ($csv in $hits) {
    $fileName = $csv.Name

    # Extract first 5 digits from file name
    $first5 = $fileName.Substring(0, [Math]::Min(5, $fileName.Length))
    $num = 0
    if ($first5 -match '^\d{5}$') {
      $num = [int]$first5
    }

    $label = if ($num -lt 21838) { 0 } else { 1 }

    $rows += [PSCustomObject]@{
      file  = $fileName
      label = $label
    }
  }
  return $rows
}

Write-Host "Scanning $DataDir for .csv files..."
$rows = Get-LabelRows -Root $DataDir
Write-Host "  Found $($rows.Count) .csv files."

# Optional: de-duplicate on 'file' (keeps first occurrence)
$rows = $rows | Sort-Object file -Unique

# Write CSV exactly as: file,label
$rows | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8

Write-Host "Wrote $($rows.Count) rows to $OutputCsv"
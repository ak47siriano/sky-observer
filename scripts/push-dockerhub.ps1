param(
    [Parameter(Mandatory = $true)]
    [string]$DockerHubUser,

    [string]$ImageName = "sky-observer",
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
$FullImage = "$DockerHubUser/$ImageName`:$Tag"
$TimestampTag = "$DockerHubUser/$ImageName`:" + (Get-Date -Format "yyyyMMdd-HHmm")

Write-Host "=== 1/3 Build immagine unificata (Frontend + Backend AI): $FullImage ===" -ForegroundColor Cyan
docker build -t $FullImage -t $TimestampTag -f Dockerfile .

Write-Host "=== 2/3 Push versione :$Tag su Docker Hub ===" -ForegroundColor Cyan
docker push $FullImage

Write-Host "=== 3/3 Push versione storica $TimestampTag su Docker Hub ===" -ForegroundColor Cyan
docker push $TimestampTag

Write-Host "Completato! Immagine pubblicata su Docker Hub: $FullImage" -ForegroundColor Green

<div align="center">

<p>
  <img src="docs/assets/logo-simple.svg" alt="Sky Observer — Logo Modalità Semplice" width="108" height="108" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/assets/logo-expert.svg" alt="Sky Observer — Logo Modalità Esperto" width="108" height="108" />
</p>

# Sky Observer

**Osservatorio cielo AI local-first, mappa georeferenziata di proiezione nuvolosa e nowcasting a breve termine con optical flow da smartphone, camera USB/RTSP, ESP32-CAM o AllSky.**

*Funziona al 100% offline su CPU fin dal primo avvio grazie al motore fisico radiometrico calibrato (`radiometric-physics-v1`), con selettore istantaneo **`Semplice` ↔ `Esperto`** (e icona adattiva Doppler Scope), backend neurali PyTorch opzionali (`DeepLabV3`, `DINOv3` linear probe, `CloudSegUNet`), nowcasting confrontato onestamente contro la persistenza, metriche Prometheus e integrazione Home Assistant.*

[![CI](https://github.com/ak47siriano/sky-observer/actions/workflows/ci.yml/badge.svg)](https://github.com/ak47siriano/sky-observer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Latest Release](https://img.shields.io/github/v/release/ak47siriano/sky-observer?include_prereleases)](https://github.com/ak47siriano/sky-observer/releases)

**[English](README.md)** · **[Italiano](README.it.md)**

</div>

---

## Anteprima Live

<p align="center">
  <img src="docs/assets/sky-observer-demo.gif" alt="Sky Observer — Cielo Originale, Segmentazione Nuvolosa, Optical Flow e Mappa Nuvole Proiettata" width="100%" />
</p>

> *Cattura reale che alterna **Cielo Originale -> Segmentazione Calibrata -> Vettori di Moto Optical Flow -> Confronto Swipe -> Mappa Proiezione Nuvole + Radar Precipitazioni RainViewer opzionale**.*

---

## Funzionalità Principali

| Area | Implementazione |
| :--- | :--- |
| **Doppia Interfaccia (`Semplice` ↔ `Esperto`)** | Switch persistente con 1 clic nell'intestazione e logo **Doppler Scope** adattivo: la **Modalità Semplice** offre verdetto visivo del cielo, mappa nuvole/pioggia e configurazione rapida; la **Modalità Esperto** sblocca telemetria METAR/SYNOP, confronto swipe, vettori di moto, previsioni e AI Lab. |
| **Segmentazione Zero-GPU Predefinita** | `radiometric-physics-v1` (CIE-Lab + rapporto Rosso/Blu + normalizzazione AE/AWB + guardie anti-abbagliamento solare e base scura) gira su **PC solo CPU da 4 GB RAM** senza scaricare pesi; supporta la proiezione ridotta a **3 classi** (`clear_sky` / `cloud` / `thick_cloud` + `255` esclusione non-cielo). |
| **Backend Neurali Plug-in** | Supporto opzionale a **`DeepLabV3-MobileNetV3`** (diurno, BSD-3-Clause), **`DINOv3-LinearProbe`** (giorno/notte su 50–100 frame locali via [`scripts/train_dinov3_probe.py`](scripts/train_dinov3_probe.py)) e `CloudSegUNet`. I checkpoint `[baseline-default]` sono protetti da pulizie disco e pruning. |
| **Classificazione Macro (5 Classi SWIMCAT)** | Accanto alle etichette dettagliate e ai tag di stato cielo (`contrails`, `rain`, `snow`, `fog`), espone 5 macro-classi realistiche da singola camera (`cumuliform_low`, `stratiform_low`, `mid_level`, `high_cirriform`, `convective_storm` + `clear_sky` / `unknown`). |
| **Nowcasting Onesto & Skill Score** | Advezione semi-Lagrangiana con optical flow + ensemble perturbato (`cover_p10`–`cover_p90`, `fov_exit_fraction`) confrontato esplicitamente contro la **Persistenza** (`skill = 1 - MAE_modello / MAE_persistenza`) con avviso `low_reliability` sugli orizzonti non affidabili. |
| **Mappa Proiezione Nuvole & Stereo** | Proietta la maschera nuvolosa su Leaflet (con supporto offline a `.pmtiles` locali), cutoff zenitale configurabile (`<= 75°`, mai sotto i `15°` di elevazione), sfumatura di incertezza e endpoint **triangolazione stereo multi-camera** (`POST /api/camera/stereo/triangulate`). Integra opzionalmente **RainViewer** (frame radar passati, `z <= 7`, Universal Blue) o un server self-hosted **LibreWXR**. |
| **Auto-Train Anti-Drift Rigoroso** | Il Golden Set bloccato non entra mai nel training, al massimo il **50% del batch** può contenere pseudo-label e un nuovo checkpoint viene promosso solo se migliora senza regredire oltre soglia in nessuna categoria di cielo. |
| **Sicurezza & Smart-Home / Prometheus** | Autenticazione opzionale con token (`SKYOBSERVER_API_TOKEN`) su `/api/*`, `/ws/live` e `/metrics`, conferma server-side sulle azioni distruttive, filtro SSRF, quota disco (`max_storage_gb`), REST `/api/sky-state`, MQTT Auto-Discovery per Home Assistant e scrape **Prometheus `/metrics`**. |

---

## Avvio Rapido (Quick Start)

Sky Observer utilizza un **unico `Dockerfile` unificato** (FastAPI serve direttamente la PWA React pre-compilata sulla porta **`8080`**), wheel **PyTorch solo CPU** di default e **modalità Offline-First attiva di default (`SKYOBSERVER_OFFLINE_MODE=true`)**: zero richieste verso l'esterno finché non abiliti esplicitamente la modalità Ibrida.

### Opzione 1: Docker Compose (Consigliata — Immagine Unificata CPU)
```bash
cp .env.example .env
docker compose up -d --build
```
Apri **`http://localhost:8080`** (esposto solo su `127.0.0.1` di default).

- **Sorgenti camera in Docker standard**: supporta solo **sorgenti di rete** (`http_snapshot` / IP Webcam, `mjpeg_stream`, `rtsp`, `web_stream`, o una cartella `allsky_folder` montata).
- **Webcam USB / Iriun su Linux**:
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.usb.yml up -d
  ```
  *(Su Windows/macOS Docker Desktop non passa nativamente `/dev/video*`: usa uno stream di rete oppure l'installazione nativa sotto.)*
- **GPU NVIDIA (CUDA 12.4)**:
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
  ```
- **LAN & PWA Mobile in HTTPS**: i Service Worker e l'installazione PWA su smartphone richiedono HTTPS sugli IP LAN. Imposta `SKYOBSERVER_API_TOKEN`, `SKYOBSERVER_HOST=0.0.0.0` e `SKYOBSERVER_LAN_HOST=<tuo-ip-lan>` in `.env` e avvia con Caddy HTTPS (oppure usa **mkcert** / **Tailscale**, vedi [`docs/INSTALLATION.md`](docs/INSTALLATION.md)):
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.https.yml up -d
  ```

### Opzione 2: Installazione Nativa (Linux / macOS — Bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r backend/requirements.txt

cd frontend && npm ci && npm run build && cd ..

PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8080
```

### Opzione 3: Installazione Nativa (Windows — PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r backend/requirements.txt

cd frontend; npm ci; npm run build; cd ..

$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

---

## Benchmark Riproducibili (vs Baseline Naive)

Tutti i numeri seguenti sono misurati direttamente da `PYTHONPATH=backend python scripts/evaluate.py` con **split cronologici a giornate intere** e baseline esplicite (zero metriche inventate; golden set locali e dataset pubblici `SWIMSEG`/`SWINySEG`/`WSISEG` in `data/datasets/` vengono valutati automaticamente):

| Task | Dataset | Modello vs Baseline | Risultato Misurato |
| :--- | :--- | :--- | :--- |
| **Segmentazione** | `bundled_synthetic_sky_18f` (`n=18`) | `radiometric-physics-v1` vs R/B Fisso vs Otsu | Physics mIoU **`0.8725`** (Cover MAE **`5.02%`**) · R/B Fisso `0.8628` (`7.09%`) · Otsu `0.8375` (`7.78%`) |
| **Classificazione** | `bundled_synthetic_sky_18f` (giorno held-out `n=6`) | `physical-heuristic-v1` vs kNN 12-feat vs Maggioranza | Euristica Acc **`0.6667`** (F1 `0.5714`) · kNN **`0.8333`** (F1 `0.7778`) · Maggioranza `0.1667` |
| **Nowcast (+5 min)** | `bundled_panoramic_cloud_front_14f` (`n=11`) | Pers vs Lineare vs OF vs Ibrido | Pers `12.59%` · Lin `5.92%` (`+0.53`) · OF `9.47%` (`+0.25`) · Ibrido `6.49%` (`+0.48`) |
| **Nowcast (+15 min)** | `bundled_panoramic_cloud_front_14f` (`n=7`) | Pers vs Lineare vs OF vs Ibrido | Pers `41.27%` · Lin `16.77%` (`+0.59`) · OF `33.83%` (`+0.18`) · Ibrido `26.37%` (`+0.36`) |
| **Nowcast (+20 min)** | `bundled_panoramic_cloud_front_14f` (`n=5`) | Pers vs Lineare vs OF vs Ibrido | Pers `54.58%` · Lin `23.69%` (`+0.57`) · OF `43.98%` (`+0.19`) · Ibrido `37.66%` (`+0.31`) |

Dettagli completi (metriche stratificate per condizione e verifica eventi su finestra futura oggettiva) in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Integrazioni: Home Assistant & Prometheus

- **Home Assistant REST**: interroga `GET http://<ip-lan>:8080/api/sky-state` (o attiva `mqtt:` in `config/config.yaml` con `pip install -e ".[mqtt]"` per l'Auto-Discovery automatica).
- **Prometheus (`/metrics`)**: endpoint nativo in formato testuale Prometheus (`sky_observer_up`, `sky_observer_camera_connected`, `sky_observer_observations_total`, `sky_observer_cloud_coverage_pct`, `sky_observer_quality_score`, `sky_observer_inference_ms`, `sky_observer_storage_used_mb`, `sky_observer_active_models_count`). Se `SKYOBSERVER_API_TOKEN` è impostato, passa `Authorization: Bearer <token>`.
- **Verifica METAR**: `GET /api/verification/metar` confronta l'osservazione locale con i bollettini METAR più vicini (disabilitato automaticamente in modalità Offline).

---

## Limitazioni Note

Sky Observer documenta in modo trasparente i limiti fisici di una stazione basata su camera consumer:

1. **Incertezza della Quota Base Nubi (Monoculare)**: una singola camera non può misurare direttamente l'altitudine delle nubi. L'errore sulla quota stimata (`LCL = 125 * (T - Td)` o valore manuale) si trasferisce linearmente sulla distanza proiettata in mappa. Per ottenere quote reali servono almeno due camere distanziate con l'endpoint di triangolazione stereo `POST /api/camera/stereo/triangulate`.
2. **Distorsione all'Orizzonte (`tan(zenith)`)**: la proiezione diverge avvicinandosi all'orizzonte; il sistema esclude sempre i pixel oltre `75°` di zenit (`< 15°` di elevazione) e sfuma progressivamente l'opacità con l'incertezza.
3. **Il Sistema NON È un Radar Meteorologico**: la mappa mostra la **proiezione della copertura nuvolosa ottica** osservata dalla camera, non l'intensità della precipitazione. Le precipitazioni sono visualizzate solo attivando il layer esterno opzionale **RainViewer** (o un proprio server **LibreWXR**).
4. **Orizzonte di Nowcasting Limitato dal FOV**: con un obiettivo smartphone standard (~70°), le nubi attraversano l'intera inquadratura in 5–15 minuti. Il nowcasting solo-camera è quindi limitato a `+5/+15/+20 min` (con bandiera `low_reliability` se lo skill score non batte la persistenza); orizzonti `+30/+60 min` richiedono i dati NWP di Open-Meteo in modalità Ibrida.
5. **Notte, Vetri Interni e Smartphone AE/AWB**: di notte il rilevamento dipende da bagliore urbano o visibilità stellare; attraverso vetri interni i riflessi possono introdurre artefatti (segnalati dal QC `window_reflection`). La normalizzazione AE/AWB e la correzione distorsione lente (`lens_k1/k2/p1/p2`) riducono ma non eliminano gli artefatti dei sensori consumer.
6. **Rischio Drift nell'Auto-Train**: il self-training non supervisionato può degradare nel tempo; per mitigarlo il Golden Set è escluso dal training, le pseudo-label non superano mai il 50% del batch e il gate di promozione scarta qualsiasi checkpoint che peggiori oltre soglia anche su una sola categoria di cielo.

---

## Documentazione Completa

- [`docs/INSTALLATION.md`](docs/INSTALLATION.md) — Profili hardware, Docker/nativo, HTTPS LAN e protezione smartphone
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Architettura di sistema e pipeline visiva
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — Metriche riproducibili, baseline e riferimenti scientifici
- [`docs/API.md`](docs/API.md) — Riferimento completo API REST e WebSocket
- [`docs/DATASETS.md`](docs/DATASETS.md) — Dataset pubblici, licenze e tabelle di conversione classi
- [`SECURITY.md`](SECURITY.md) — Modello di minaccia, token API, guardie SSRF e gestione credenziali
- [`CONTRIBUTING.md`](CONTRIBUTING.md) & [`CHANGELOG.md`](CHANGELOG.md) — Linee guida contributi e storico versioni

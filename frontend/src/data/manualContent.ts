export type ManualSectionId =
  | "dashboard"
  | "radar"
  | "forecast"
  | "events"
  | "ai"
  | "settings"
  | "formulas";

export interface ManualParamCard {
  name: string;
  badge?: string;
  what: string;
  how: string;
  tip?: string;
}

export interface ManualSection {
  id: ManualSectionId;
  title: string;
  shortTitle: string;
  subtitle: string;
  intro: string;
  quickSteps?: string[];
  params: ManualParamCard[];
}

export const MANUAL_SECTIONS_IT: ManualSection[] = [
  {
    id: "dashboard",
    title: "1. Dashboard & Stato del Cielo",
    shortTitle: "Dashboard & Cielo",
    subtitle: "Vista live della camera, segmentazione nuvolosa, parametri istantanei e maschera di esclusione.",
    intro:
      "La Dashboard è la sala di controllo principale: mostra l'ultima foto catturata dalla tua camera, la mappa colorata delle nubi riconosciute dall'AI e tutti gli indicatori istantanei del cielo.",
    quickSteps: [
      "Usa i pulsanti sopra la foto per passare tra Ultima Foto, Segmentazione (colori delle nubi), Confronta Swipe, Vettori Moto, Timelapse ed Escludi Zone.",
      "Disegna con 'Escludi Zone' sopra alberi, tetti, mare, edifici o loghi: i pixel esclusi non entreranno mai nel calcolo della nuvolosità.",
      "Attiva 'Orizzonte 3D' per verificare visivamente che la linea dell'orizzonte (0°) coincida con l'orizzonte reale della tua webcam.",
    ],
    params: [
      {
        name: "Copertura Nuvolosa (%)",
        badge: "0% – 100%",
        what: "Percentuale di cielo effettivamente coperta da nuvole (escludendo le zone mascherate come edifici, alberi o mare).",
        how: "Il modello CloudSegUNet-Mobile classifica ogni pixel di cielo in 6 categorie (Sereno, Sottili, Dense, Scure, Foschia, Incerto). I pixel nuvolosi vengono poi pesati con la correzione geometrica 3D (angolo solido) così le nubi vicine all'orizzonte non falsano la percentuale.",
        tip: "Se hai una webcam molto inclinata, abilita sempre la Correzione Tilt 3D per una percentuale reale.",
      },
      {
        name: "Nubi Sottili / Dense / Scure / Foschia",
        badge: "Classi Pixel",
        what: "Scomposizione della copertura totale in base allo spessore e alla luminosità delle nubi.",
        how: "Sottili = velature/cirri semi-trasparenti ad alta luminosità; Dense = cumuli/stratocumuli bianchi opachi; Scure = basi spesse e scure tipiche di pioggia/temporale; Foschia = velo lattiginoso vicino all'orizzonte.",
        tip: "Un aumento rapido delle 'Nubi Scure' sopra il 20% è il primo segnale locale di rovescio o temporale in avvicinamento.",
      },
      {
        name: "Tipo di Nube Dominante",
        badge: "11 Generi WMO",
        what: "Classificazione meteorologica della formazione prevalente (es. Cumulus, Altocumulus, Cirrus, Stratus, Cumulonimbus).",
        how: "Il modello CloudTypeClassifier-Fusion combina l'immagine RGB (224×224) con 12 caratteristiche fisiche (luminosità, indice di blu, contrasto, texture GLCM e frazioni nuvolose). Sotto la soglia di confidenza minima mostra 'Sconosciuto' per evitare stime casuali.",
      },
      {
        name: "Moto Nubi (px/s e direzione)",
        badge: "Optical Flow",
        what: "Velocità apparente e direzione in cui si stanno spostando le nubi nell'immagine.",
        how: "Confronta due fotogrammi consecutivi con Flusso Ottico Denso (Farnebäck) calcolato solo sui pixel nuvolosi validi. La direzione in gradi (0°=Nord, 90°=Est, 180°=Sud, 270°=Ovest) viene corretta usando l'Azimuth della camera.",
        tip: "Se la camera è ferma e le nubi sono molto uniformi (cielo tutto bianco), il moto può segnare 0.0 px/s perché mancano bordi tracciabili.",
      },
      {
        name: "Luminosità (L*) & Indice Cielo Blu (BSI)",
        badge: "Colore Cielo",
        what: "Misura quanto è luminoso il cielo (scala percettiva CIE L* da 0=buio a 100=molto luminoso) e quanto è puro il blu.",
        how: "Calcolati sullo spazio colore LAB e sul rapporto (Blu - Rosso) / (Blu + Rosso) dei soli pixel di cielo visibile.",
        tip: "Un calo improvviso di L* durante il giorno indica l'arrivo di una copertura spessa o di un fronte temporalesco.",
      },
      {
        name: "Qualità Frame (%)",
        badge: "0% – 100%",
        what: "Indica se la foto catturata è nitida e affidabile per l'analisi AI.",
        how: "Controlla automaticamente luminosità media (scarta frame completamente neri o sovraesposti), nitidezza (varianza Laplaciana) e area di cielo valida.",
        tip: "I frame sotto la soglia minima di qualità vengono marcati come non utilizzabili e non sporcano le previsioni né l'Auto-Train.",
      },
    ],
  },
  {
    id: "radar",
    title: "2. Mappa Radar 3D & Geometria",
    shortTitle: "Mappa Radar 3D",
    subtitle: "Proiezione geografica delle nubi locali sulla mappa, calibrazione Tilt 3D, quota nubi dinamica e layer online.",
    intro:
      "La Mappa Radar trasforma ciò che vede la tua camera in una vera mappa meteorologica georeferenziata, sovrapponendo le nubi rilevate dalla tua webcam ai dati regionali Open-Meteo e al radar pioggia RainViewer.",
    quickSteps: [
      "Posiziona il pin sulla tua località (Cerca città, Usa il mio GPS, oppure clicca 'Sposta Pin' e poi sulla mappa).",
      "Apri 'Tilt Camera' in alto a destra per allineare Inclinazione (0°–110°), Direzione Bussola (Azimuth 0°–360°) e FOV dell'obiettivo.",
      "Usa il dock in vetro in basso a destra sulla mappa per cambiare stile (Scuro/Stradale/Satellite), attivare/disattivare i layer e regolare l'opacità delle nubi.",
      "Usa la barra temporale in basso per rivedere lo storico delle nubi sul radar insieme alla mini-foto PiP di quell'istante.",
    ],
    params: [
      {
        name: "Inclinazione dalla Verticale (Tilt 0°–110°)",
        badge: "Geometria 3D",
        what: "Indica quanto è inclinata la camera rispetto allo zenit (cielo esattamente sopra la testa).",
        how: "0° = camera puntata dritta verso l'alto (zenit, El 90°); 45° = inclinata a metà (El 45°); 90° = perfettamente orizzontale verso l'orizzonte (El 0°); 95°–110° = webcam panoramica montata in alto che guarda leggermente verso il basso (El fino a -20°).",
        tip: "Apri l'anteprima camera nel calibratore Tilt e regola il cursore finché la linea rossa dell'orizzonte (0°) coincide con l'orizzonte reale della foto.",
      },
      {
        name: "Azimuth Bussola (0°–360°) & Roll",
        badge: "Orientamento",
        what: "La direzione cardinale verso cui punta l'obiettivo (0°=Nord, 90°=Est, 180°=Sud, 270°=Ovest) e l'eventuale rotazione laterale (Roll).",
        how: "Ruota il cono visivo (FOV) sulla mappa geografica e allinea i vettori di moto delle nubi ai punti cardinali reali.",
      },
      {
        name: "Quota Nubi Dinamica (m)",
        badge: "AUTO Dinamica",
        what: "L'altezza stimata della base delle nuvole dal suolo, usata per calcolare a quanti chilometri di distanza si trovano le nubi che vedi.",
        how: "In modalità AUTO si aggiorna da sola: (1) Online usa una media pesata degli strati nuvolosi Open-Meteo (Basse = base LCL calcolata da Temperatura e Umidità con formula di Espy 125×(T−Td), Medie ≈ 4000m, Alte/Cirri ≈ 8500m); (2) Offline stima la quota direttamente dall'immagine in base al tipo di nube riconosciuto dall'AI.",
        tip: "Lascia il campo vuoto su AUTO per adattare automaticamente la scala del radar al meteo reale. Inserisci un numero manuale solo se vuoi forzare una quota fissa.",
      },
      {
        name: "Proiezione Volumetrica Nubi (512×512)",
        badge: "Rendering HD",
        what: "La nuvolosità reale della tua camera disegnata direttamente sopra le strade e il territorio nella mappa.",
        how: "Per ogni pixel del piano radar (a quota H), il motore geometrico calcola il raggio 3D verso la camera, campiona le densità nuvolose continue con interpolazione bilineare e applica uno shading volumetrico morbido con sfumatura ai bordi.",
      },
      {
        name: "Velocità al Suolo 3D (km/h) & Traiettoria",
        badge: "+5 / +15 / +30m",
        what: "Stima della velocità reale di spostamento delle nubi in km/h e del loro percorso previsto nei prossimi 30 minuti.",
        how: "Converte la velocità angolare (°/s) misurata dall'Optical Flow in velocità lineare al suolo moltiplicandola per la quota nubi: v = ω(rad/s) × Quota(m) × 3,6.",
      },
      {
        name: "Multi-Camera 360°",
        badge: "Fusione Radar",
        what: "Permette di collegare una seconda camera puntata in direzione opposta (es. Nord + Sud) e vedere entrambi i coni fusi sulla stessa mappa.",
        how: "Configurabile in Impostazioni → Multi-Camera 360°: ogni camera secondaria viene proiettata sul medesimo piano geografico con la propria bussola e inclinazione.",
      },
    ],
  },
  {
    id: "forecast",
    title: "3. Previsioni, Andamenti & Timelapse",
    shortTitle: "Previsioni & Dati",
    subtitle: "Nowcasting a breve termine (+5/+15/+30/+60m), verifica accuratezza MAE, esportazione CSV/JSON e Timelapse HUD.",
    intro:
      "Questa sezione raccoglie le previsioni a breve termine basate sull'evoluzione visiva del cielo, i grafici storici, la verifica automatica dell'errore dei modelli e gli strumenti di esportazione dati e video.",
    quickSteps: [
      "Controlla gli orizzonti +5m, +15m, +30m e +60m per capire se il cielo sta schiarendo o si sta coprendo.",
      "Guarda il grafico 'Andamento Errore Previsioni (MAE)' per verificare se l'AI sta diventando più precisa dopo l'Auto-Train.",
      "Usa 'Esporta CSV' o 'Esporta JSON' per analizzare tutta la serie storica in Excel o Python.",
    ],
    params: [
      {
        name: "Previsione Copertura (+5m / +15m / +30m / +60m)",
        badge: "Nowcasting",
        what: "Percentuale di nuvolosità stimata nei prossimi 5, 15, 30 e 60 minuti.",
        how: "Combina il modello neurale TemporalSkyConvGRU (che osserva la sequenza degli ultimi fotogrammi), il trend lineare recente della copertura e, se in modalità Ibrido Online, la tendenza a 15 minuti di Open-Meteo.",
      },
      {
        name: "Probabilità Schiarita / Aumento / Coperto",
        badge: "0% – 100%",
        what: "Probabilità che il cielo diventi più sereno, più nuvoloso oppure completamente coperto (>80%) nell'orizzonte scelto.",
        how: "Output probabilistico (sigmoide) delle teste del modello temporale integrato con il tasso di variazione attuale (%/min) e l'avvezione rilevata.",
      },
      {
        name: "Errore MAE (Mean Absolute Error)",
        badge: "Verifica Reale",
        what: "Misura di quanti punti percentuali sbaglia in media la previsione rispetto al cielo reale osservato quando arriva l'ora prevista.",
        how: "Quando trascorrono 5, 15, 30 o 60 minuti da una previsione, il sistema confronta automaticamente |Copertura Prevista − Copertura Reale|. Più è vicino a 0%, più il modello è preciso!",
        tip: "Un MAE sotto il 10–15% è ottimo per un nowcasting da singola camera locale.",
      },
      {
        name: "Timelapse con HUD Meteo & Riassunto Giornaliero",
        badge: "MP4 + HUD",
        what: "Crea video MP4 velocizzati del tuo cielo con sovraimpressi data/ora, % copertura, tipo di nube, direzione moto e temperatura.",
        how: "Generabile su richiesta nella scheda Timelapse; inoltre, ogni sera dopo il tramonto, il sistema genera automaticamente il Riassunto Giornaliero (YYYY-MM-DD_daily.mp4).",
      },
    ],
  },
  {
    id: "events",
    title: "4. Eventi, Avvisi & Feedback AI",
    shortTitle: "Eventi & Avvisi",
    subtitle: "Rilevamento automatico di cambi meteo rapidi, oscuramento, temporali e apprendimento dal tuo feedback.",
    intro:
      "Sky Observer monitora continuamente il cielo e genera un avviso con foto allegata ogni volta che rileva un cambiamento significativo (arrivo temporale, schiarita rapida, addensamento, ostruzione camera).",
    quickSteps: [
      "Per ogni avviso puoi cliccare 'Conferma' (era reale) oppure 'Falso Allarme' (era sbagliato).",
      "Il tuo feedback viene salvato e usato dal ciclo di Auto-Train Locale per insegnare all'AI a non ripetere falsi allarmi sul tuo cielo.",
    ],
    params: [
      {
        name: "Severità Avviso (Info / Attenzione / Critico)",
        badge: "Classificazione",
        what: "Livello di importanza dell'evento rilevato.",
        how: "Info = schiarita o cambio lieve; Attenzione = rapido aumento nuvolosità o oscuramento; Critico = struttura di tipo temporalesco o oscuramento intenso con nubi scure elevate.",
      },
      {
        name: "Stato Feedback (In attesa / Confermato / Rifiutato)",
        badge: "Human-in-the-Loop",
        what: "Indica se hai validato manualmente l'avviso.",
        how: "Gli eventi confermati diventano esempi positivi ad alta priorità nel dataset locale di addestramento; quelli rifiutati vengono nascosti dalla vista principale e usati come contro-esempi.",
      },
    ],
  },
  {
    id: "ai",
    title: "5. AI Lab, 4 Modelli & Hardware GPU",
    shortTitle: "AI Lab & Modelli",
    subtitle: "Addestramento locale, Auto-Train, gestione versioni modelli, metriche di validazione e accelerazione GPU/CPU.",
    intro:
      "AI Lab ti permette di controllare il cervello neurale locale di Sky Observer: scegliere l'hardware (CPU o GPU), avviare il fine-tuning sui tuoi cieli, gestire le versioni dei 4 modelli PyTorch e pulire i vecchi checkpoint.",
    quickSteps: [
      "Scegli il Profilo Hardware adatto al tuo PC (Low-End, Bilanciato, High-End, Ultra) oppure lascia 'Auto'.",
      "Attiva l'Auto-Train Locale per far migliorare automaticamente i modelli usando le foto del tuo cielo.",
      "Nel Registro Modelli puoi attivare una versione precedente (Rollback) oppure usare 'Pulisci Vecchi Modelli' per liberare spazio.",
    ],
    params: [
      {
        name: "val_miou & val_pixel_acc (Modello 1 — Segmentazione)",
        badge: "0.00 – 1.00",
        what: "Precisione con cui CloudSegUNet-Mobile disegna i contorni delle nuvole.",
        how: "val_miou (Mean Intersection over Union) misura quanto i contorni predetti coincidono con quelli reali (più vicino a 1.0 = meglio). val_pixel_acc è la percentuale di pixel classificati correttamente (es. 0.88 = 88%).",
      },
      {
        name: "val_accuracy & val_macro_f1 (Modelli 2 e 4)",
        badge: "0.00 – 1.00",
        what: "Affidabilità del classificatore di Tipo Nube e del rilevatore di Eventi.",
        how: "val_accuracy è la % di risposte esatte; val_macro_f1 è una media bilanciata tra precisione e richiamo su tutte le classi (utile per capire se riconosce bene anche i cieli rari come i temporali).",
      },
      {
        name: "val_mae_pct (Modello 3 — Previsione Temporale)",
        badge: "Più basso = meglio",
        what: "Errore medio assoluto in punti percentuali della previsione di copertura nuvolosa.",
        how: "Media delle differenze |Previsto − Reale| sul set di validazione. Esempio: 12.0 significa uno scarto medio di ±12% sulla nuvolosità futura.",
      },
      {
        name: "Accelerazione Hardware (CUDA / ROCm / DirectML / MPS / CPU)",
        badge: "Multi-GPU",
        what: "Seleziona dove vengono eseguiti inferenza e addestramento PyTorch.",
        how: "Supporta NVIDIA CUDA, AMD ROCm (Linux), AMD/Intel DirectML (Windows) e Apple MPS. Su Windows con DirectML, l'inferenza live gira veloce sulla GPU AMD mentre il training automatico usa la CPU per evitare timeout TDR del driver video.",
      },
    ],
  },
  {
    id: "settings",
    title: "6. Impostazioni, Sorgenti Video & Storage",
    shortTitle: "Impostazioni",
    subtitle: "Configurazione di qualsiasi sorgente video (Webcam USB/Virtuale, Camera IP/RTSP, Stream Web Pubblico, File Locali), modalità Offline/Online e pulizia dati.",
    intro:
      "Qui configuri da dove Sky Observer riceve le immagini del cielo (qualunque tipo di camera o stream), la frequenza di cattura, le camere secondarie per il radar 360° e la gestione dello spazio su disco.",
    quickSteps: [
      "Scegli il Tipo di Sorgente Input: Webcam Locale / USB / Camera Virtuale (es. webcam PC, Iriun, OBS), Snapshot HTTP / MJPEG, Stream RTSP, Stream Web Pubblico (HLS .m3u8 o SkylineWebcams) oppure Importazione Manuale di foto/video.",
      "Premi 'Test' per verificare subito la connessione e poi 'Salva Tutte le Impostazioni'.",
      "Usa 'Solo Offline' in alto se vuoi bloccare al 100% qualsiasi connessione verso internet (Open-Meteo/RainViewer/Stream pubblici).",
    ],
    params: [
      {
        name: "Stream Web Pubblico (web_stream)",
        badge: "HLS / SkylineWebcams",
        what: "Permette di usare qualsiasi webcam pubblica live su internet come sorgente cielo.",
        how: "Incolla l'URL diretto di una playlist .m3u8 oppure il link di una pagina SkylineWebcams: Sky Observer estrae automaticamente lo stream video live, scarta eventuali frame neri iniziali e analizza il cielo in tempo reale (richiede modalità Ibrido Online).",
      },
      {
        name: "Intervallo Cattura (secondi)",
        badge: "Default: 5s",
        what: "Ogni quanti secondi viene acquisito e analizzato un nuovo fotogramma.",
        how: "Valori tra 5s e 15s sono ideali per tracciare bene il moto delle nubi (Optical Flow) senza appesantire troppo il disco.",
      },
      {
        name: "Centro Pulizia Dati & Reset",
        badge: "Storage SQLite",
        what: "Permette di controllare lo spazio occupato da foto, maschere, timelapse, modelli e database.",
        how: "Puoi cancellare selettivamente solo le foto pesanti mantenendo intatti i grafici numerici e i modelli AI, oppure eseguire una pulizia completa con compattazione SQLite VACUUM.",
      },
    ],
  },
  {
    id: "formulas",
    title: "7. Riepilogo Formule & Calcoli Chiave",
    shortTitle: "Formule & Calcoli",
    subtitle: "Tutte le formule fisiche e matematiche usate da Sky Observer spiegate in modo semplice.",
    intro:
      "Sky Observer non tira a indovinare: combina reti neurali locali con equazioni reali di meteorologia, ottica fotografica e astronomia solare. Ecco il riepilogo immediato di ogni calcolo.",
    params: [
      {
        name: "Base Nubi LCL (Formula di Espy + Magnus-Tetens)",
        badge: "Termodinamica",
        what: "Stima l'altezza (in metri) a cui il vapore acqueo condensa formando la base delle nubi basse.",
        how: "1) Calcola il Punto di Rugiada Td da Temperatura (T) e Umidità Relativa (RH%). 2) Applica H_LCL ≈ 125 × (T − Td) metri. Esempio: con T = 29°C e Td = 20,7°C → 125 × 8,3 ≈ 1040 m.",
      },
      {
        name: "Quota Dinamica Multi-Strato",
        badge: "Open-Meteo NWP",
        what: "Adatta la quota proiettata sul radar quando ci sono nubi medie o alte (cirri) oltre a quelle basse.",
        how: "Media pesata: (Basse% × H_LCL + Medie% × 4000m + Alte% × 8500m) / (Basse% + Medie% + Alte%). Se offline, usa la quota tipica del genere nuvoloso riconosciuto dall'AI.",
      },
      {
        name: "Distanza a Terra di una Nube sul Radar",
        badge: "Trigonometria 3D",
        what: "Trasforma la posizione di un pixel nella foto nella sua posizione geografica (Latitudine, Longitudine).",
        how: "Dato l'angolo di elevazione α del pixel rispetto all'orizzonte e la quota nuvolosa H, la distanza orizzontale è D = H / tan(α). Più la nube è vicina all'orizzonte (α piccolo), più è lontana in km.",
      },
      {
        name: "Posizione Solare & Protezione Abbagliamento",
        badge: "Astronomia NOAA",
        what: "Calcola l'esatta posizione del Sole (Elevazione e Azimuth) per la tua località e ora UTC.",
        how: "Usa le equazioni astronomiche solari (declinazione solare + equazione del tempo) per proiettare il disco solare sulla foto e distinguere il bagliore del sole dalle nuvole reali.",
      },
      {
        name: "Velocità Reale delle Nubi (km/h)",
        badge: "Cinematica",
        what: "Stima la velocità del vento in quota osservando il movimento delle nubi.",
        how: "Velocità angolare ω (°/s) = (pixel/s × FOV°) / Larghezza_px. Poi Velocità al suolo = ω(rad/s) × Quota_Nubi(m) × 3,6.",
      },
    ],
  },
];

export const MANUAL_SECTIONS_EN: ManualSection[] = [
  {
    id: "dashboard",
    title: "1. Dashboard & Sky State",
    shortTitle: "Dashboard & Sky",
    subtitle: "Live camera stage, cloud pixel segmentation, instant sky telemetry, and exclusion mask painter.",
    intro:
      "The Dashboard is your main observatory view: it shows the latest captured sky frame, the AI-segmented cloud map, and all real-time atmospheric indicators.",
    quickSteps: [
      "Switch between Raw Photo, Segmentation, Swipe Compare, Motion Vectors, Timelapse, and Exclude Zones above the sky stage.",
      "Paint over trees, roofs, sea, buildings, or logos in 'Exclude Zones' so non-sky pixels never bias cloud coverage.",
      "Toggle '3D Horizon' to verify that the mathematical 0° horizon aligns with your real camera horizon.",
    ],
    params: [
      {
        name: "Cloud Coverage (%)",
        badge: "0% – 100%",
        what: "Percentage of visible sky covered by clouds (excluding masked areas).",
        how: "CloudSegUNet-Mobile classifies every sky pixel into 6 classes (Clear, Thin, Dense, Dark, Haze, Uncertain). Pixels are weighted by 3D solid-angle geometry so perspective near the horizon never inflates coverage.",
      },
      {
        name: "Thin / Dense / Dark Clouds & Haze",
        badge: "Pixel Classes",
        what: "Breakdown of total cloud cover by optical thickness and brightness.",
        how: "Thin = bright translucent cirrus/veils; Dense = opaque white cumulus/stratocumulus; Dark = thick storm/rain bases; Haze = misty veil near the horizon.",
      },
      {
        name: "Dominant Cloud Type",
        badge: "11 WMO Genera",
        what: "Meteorological cloud genus classification (Cumulus, Altocumulus, Cirrus, Stratus, Cumulonimbus, etc.).",
        how: "CloudTypeClassifier-Fusion combines the 224×224 RGB image with 12 physical sky features (luminance, blue-sky index, texture contrast, cloud fractions).",
      },
      {
        name: "Cloud Motion (px/s & Direction)",
        badge: "Optical Flow",
        what: "Apparent speed and compass direction of cloud advection across the sky.",
        how: "Computed between consecutive frames via Farnebäck Dense Optical Flow over cloud pixels only, rotated by camera compass azimuth.",
      },
      {
        name: "Luminance (L*) & Frame Quality (%)",
        badge: "Sky Photometry",
        what: "Perceptual sky brightness (CIE L* 0–100) and image usability score.",
        how: "Checks mean brightness (rejecting black or blown-out frames), Laplacian sharpness, and valid sky area before allowing frames into forecasts or Auto-Train.",
      },
    ],
  },
  {
    id: "radar",
    title: "2. 3D Radar Map & Geometry",
    shortTitle: "3D Radar Map",
    subtitle: "Georeferenced volumetric cloud projection, 0°–110° camera tilt calibration, dynamic cloud altitude, and radar layers.",
    intro:
      "The Radar Map projects clouds detected by your camera directly onto a geographic Leaflet map, combining local vision with Open-Meteo regional cloud decks and RainViewer precipitation radar.",
    quickSteps: [
      "Set your station pin via City Search, GPS, or 'Move Pin' mode.",
      "Open 'Camera Tilt' to match Tilt from Vertical (0°–110°), Compass Azimuth (0°–360°), and lens FOV.",
      "Use the bottom-right floating glass dock to switch Dark/Street/Satellite basemaps, toggle layers, and adjust live Cloud Opacity.",
    ],
    params: [
      {
        name: "Tilt from Vertical (0°–110°)",
        badge: "3D Geometry",
        what: "How far the camera tilts down from straight-up zenith (0°).",
        how: "0° = zenith (El 90°); 90° = horizontal (El 0°); 95°–110° = elevated panoramic/beach webcams tilted slightly downward (El down to -20°).",
      },
      {
        name: "Dynamic Cloud Altitude (m)",
        badge: "AUTO Dynamic",
        what: "Estimated cloud-base height above ground used to scale ground distances on the radar.",
        how: "In AUTO mode it updates dynamically: Online weights Open-Meteo Low (Magnus-Tetens LCL), Mid (~4000m), and High (~8500m) cloud decks; Offline estimates altitude from the Vision AI cloud type & classes.",
      },
      {
        name: "Volumetric HD Cloud Projection (512×512)",
        badge: "Bilinear Warp",
        what: "Real-time cloud shapes from your camera projected onto the map.",
        how: "Projects denoised 4-channel float32 cloud densities via bilinear warping with soft S-curve edge feathering and altitude-consistent horizon capping.",
      },
      {
        name: "3D Ground Speed (km/h) & Trajectory",
        badge: "+5 / +15 / +30m",
        what: "Estimated real-world cloud speed and arrival cone over the next 30 minutes.",
        how: "Converts angular speed (rad/s) into ground speed: v = ω × CloudAltitude(m) × 3.6.",
      },
    ],
  },
  {
    id: "forecast",
    title: "3. Forecast, Trends & Timelapse",
    shortTitle: "Forecast & Data",
    subtitle: "Short-term nowcasting (+5/+15/+30/+60m), MAE accuracy verification, CSV/JSON export, and HUD timelapses.",
    intro:
      "Track short-term cloud evolution, verify AI accuracy against real future observations, export historical datasets, and build weather-HUD timelapses.",
    params: [
      {
        name: "Nowcast Horizons (+5m / +15m / +30m / +60m)",
        badge: "Temporal AI",
        what: "Predicted future cloud cover % and clearing/increasing/overcast probabilities.",
        how: "Fuses TemporalSkyConvGRU sequence predictions, recent coverage slope, and (in Online Hybrid mode) Open-Meteo 15-minute trends.",
      },
      {
        name: "Forecast MAE (Mean Absolute Error)",
        badge: "Verification",
        what: "Average percentage-point error between past predictions and what actually happened.",
        how: "Automatically computed as |Predicted − Actual| once each forecast horizon matures. Lower is better (< 15% is strong for local camera nowcasting).",
      },
      {
        name: "HUD Timelapse & Auto Sunset Digest",
        badge: "MP4 Export",
        what: "Exports MP4 sky videos with burned-in timestamp, cloud %, genus, motion, and temperature HUD.",
        how: "Trigger anytime in Timelapse or rely on the automatic daily summary (YYYY-MM-DD_daily.mp4) generated after local sunset.",
      },
    ],
  },
  {
    id: "events",
    title: "4. Events, Alerts & AI Feedback",
    shortTitle: "Events & Alerts",
    subtitle: "Automatic sky change detection and human-in-the-loop reinforcement.",
    intro:
      "Sky Observer logs weather transitions and operational alerts with attached sky snapshots.",
    params: [
      {
        name: "Confirm / False Alarm Feedback",
        badge: "Self-Learning",
        what: "Validates or dismisses automatic alerts.",
        how: "Confirmed alerts become high-priority training samples for Location Auto-Train; rejected false alarms are hidden and used to reduce future noise.",
      },
    ],
  },
  {
    id: "ai",
    title: "5. AI Lab, 4 Models & Hardware",
    shortTitle: "AI Lab & GPU",
    subtitle: "Local fine-tuning, Auto-Train, model versioning, validation metrics, and GPU/CPU profiles.",
    intro:
      "Manage the 4 local PyTorch models, select hardware performance presets, and monitor validation metrics.",
    params: [
      {
        name: "val_miou & val_pixel_acc (Segmentation)",
        badge: "0.00 – 1.00",
        what: "Contour overlap precision and pixel classification accuracy of CloudSegUNet-Mobile.",
        how: "Higher is better (closer to 1.0 = sharper cloud boundaries).",
      },
      {
        name: "val_accuracy & val_macro_f1 (Classification & Events)",
        badge: "0.00 – 1.00",
        what: "Exact match rate and class-balanced F1 score across cloud genera and weather events.",
        how: "Higher is better; macro F1 ensures rare events like storms are scored fairly.",
      },
      {
        name: "val_mae_pct (Temporal Nowcast)",
        badge: "Lower is better",
        what: "Mean Absolute Error in percentage points on held-out validation sequences.",
        how: "Lower values mean more accurate future cloud cover predictions.",
      },
      {
        name: "GPU Backends (CUDA / ROCm / DirectML / MPS / CPU)",
        badge: "Cross-Platform",
        what: "Selects the compute backend for inference and training.",
        how: "On Windows DirectML (AMD/Intel GPUs), live inference runs fast on GPU while automatic training uses CPU to prevent Windows WDDM 2-second TDR display resets.",
      },
    ],
  },
  {
    id: "settings",
    title: "6. Settings, Video Sources & Storage",
    shortTitle: "Settings",
    subtitle: "Configure any video input (USB/Virtual Webcam, IP/RTSP Camera, Public Web Stream HLS/SkylineWebcams, Local Files), Offline lock, and storage cleanup.",
    intro:
      "Configure your video input source, capture interval, 360° secondary cameras, AI compute backend, and local disk retention.",
    params: [
      {
        name: "Public Web Live Stream (web_stream)",
        badge: "HLS / SkylineWebcams",
        what: "Use public live webcams as a sky source.",
        how: "Paste a direct .m3u8 URL or SkylineWebcams page URL (Online Hybrid mode required). Initial black H.264 warmup frames are automatically skipped.",
      },
      {
        name: "Offline Only vs Online Hybrid",
        badge: "Privacy Lock",
        what: "Controls whether outbound weather/radar/stream requests are allowed.",
        how: "Offline Only blocks 100% of external network calls; Online Hybrid enables free zero-key Open-Meteo, RainViewer, and web streams.",
      },
    ],
  },
  {
    id: "formulas",
    title: "7. Core Formulas & Calculations",
    shortTitle: "Formulas",
    subtitle: "Plain-language reference of every physical & geometric calculation in Sky Observer.",
    intro:
      "Sky Observer combines local neural networks with real meteorology, pinhole optics, and NOAA solar astronomy.",
    params: [
      {
        name: "Cloud Base LCL (Espy + Magnus-Tetens)",
        badge: "Thermodynamics",
        what: "Estimates low-cloud condensation altitude from surface temperature and humidity.",
        how: "Computes dewpoint Td from T and RH%, then H_LCL ≈ 125 × (T − Td) meters.",
      },
      {
        name: "Multi-Layer Dynamic Cloud Altitude",
        badge: "Open-Meteo NWP",
        what: "Adjusts radar projection height when mid or high clouds dominate.",
        how: "Weighted average: (Low% × H_LCL + Mid% × 4000m + High% × 8500m) / (Low% + Mid% + High%).",
      },
      {
        name: "Ground Distance on Radar",
        badge: "3D Trigonometry",
        what: "Maps each sky pixel to geographic coordinates (Lat, Lon).",
        how: "For pixel elevation angle α above the horizon and cloud altitude H: Distance D = H / tan(α).",
      },
      {
        name: "Cloud Ground Speed (km/h)",
        badge: "Kinematics",
        what: "Estimates wind speed at cloud altitude from optical flow.",
        how: "Angular speed ω = (px/s × FOV) / ImageWidth, then Ground Speed = ω(rad/s) × H(m) × 3.6.",
      },
    ],
  },
];

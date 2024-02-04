# graph-diffusion-autoencoder

TODOs
- Reconstruction
    - Paul schickt modelle an Hannah (10 und 12 Atome, conditional und unconditional)
    - Plots ein Graph aus dem Validation set und viele mögliche Reconstructions
    - Dafür Hyperparemter auch ein bisschen tunen (z. B. superconditioning)
- Treffen Wann?
    - 12 Uhr IfI

Alles 12 groß

Presentation
- Grund Idee - 4 min Paul
    - Skizze mit Encoder und Decoder und Graph und Noise Graph: Pooling, Invariant vs Equivariant
    - Noisy Graph und ratio encoding
- Datensatz vorstellen (Saturated carbon subset) - Kia 2 min
- Roadblocks 
    1. Schritt - Kia 2 min
        - Datensatz - Kia
        - Score Matching für Toy Daten - Hannah
        - Equivariantes GNN - Paul
        - Visualisieren mit den 2 Failure modes, zu hoher degree und disconnected - Kia
            - Extra Slide
    2. Schritt - Hannah 2 min
        - GPUs auf dem GWDG HPC benutzen können - Kia
        - Denoising Trainingscode - Paul
        - Langevin Dynamics für Graphen - Hannah
            - Langevin Dynamics 
    3. Schritt 3 min
        - Conditional Model implementieren - Kia
        - Logging - Hannah
        - Loss plot daneben
        - Architektur verbessern - Paul (extra slide)
            - Was wir versucht haben
            - Transformer, Mixer, GAT und ResGCN + Norm, kurze errinnerung was was ist
            - Virtual Node
    4. Schritt 2 min
        - Conditional Model trainieren - Kia & Paul
        - Sampling tunen - Hannah
            - Min sigma, max sigma
- Ergebnisse
    - Reconstructions - Hannah 2 min
    - Latent Space mit Interpretationen - Kia 3 min

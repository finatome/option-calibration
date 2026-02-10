from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

def create_layout():
    # Style constants for Pure Black Theme
    BLACK_STYLE = {"background-color": "#000000", "color": "white"}
    CARD_STYLE = {"background-color": "#000000", "color": "white", "border": "1px solid #333"}
    HEADER_STYLE = {"background-color": "#000000", "color": "white", "border-bottom": "1px solid #333"}
    INPUT_STYLE = {"background-color": "#000000", "color": "white", "border": "1px solid #444"}
    
    # Left Panel: Controls
    left_panel = dbc.Card([
        dbc.CardHeader("Configuration", style=HEADER_STYLE),
        dbc.CardBody([
            html.H5("1. Market Data Source", className="card-title"),
            dbc.RadioItems(
                options=[
                    {"label": "Synthetic Data", "value": "synthetic"},
                    {"label": "Upload File (Not Impl)", "value": "upload", "disabled": True},
                ],
                value="synthetic",
                id="data-source-radio",
                inline=True,
                style={"color": "white"}
            ),
            html.Hr(),
            
            # Synthetic Data Controls
            html.Div([
                html.H6("Synthetic Generation"),
                dbc.Row([
                    dbc.Col([html.Label("Spot (S0)"), dcc.Input(id="synth-s0", type="number", value=100.0, step=0.1, className="form-control")]),
                    dbc.Col([html.Label("Rate (r)"), dcc.Input(id="synth-r", type="number", value=0.05, step=0.01, className="form-control")])
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col([html.Label("True Kappa"), dcc.Input(id="synth-kappa", type="number", value=2.0, step=0.1, className="form-control")]),
                    dbc.Col([html.Label("True Theta"), dcc.Input(id="synth-theta", type="number", value=0.04, step=0.01, className="form-control")])
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col([html.Label("True VolVol"), dcc.Input(id="synth-xi", type="number", value=0.3, step=0.1, className="form-control")]),
                    dbc.Col([html.Label("True Rho"), dcc.Input(id="synth-rho", type="number", value=-0.5, step=0.1, className="form-control")])
                ]),
                html.Br(),
                dbc.Row([
                     dbc.Col([html.Label("Noise (%)"), dcc.Input(id="synth-noise", type="number", value=0.0, step=0.01, className="form-control")])
                ]),
                html.Br(),
                dbc.Button("Generate Market Data", id="generate-btn", color="primary", className="w-100", size="sm"),
                html.Hr(),
            ], id="synthetic-controls"),

            html.H5("2. Calibration Settings", className="card-title"),
            dcc.Dropdown(
                id="pricing-model",
                options=[
                    {'label': 'Heston Model', 'value': 'Heston'}, 
                    {'label': 'Merton Jump Diffusion', 'value': 'Merton'},
                    {'label': 'Bates Model', 'value': 'Bates'},
                    {'label': 'Black-Scholes', 'value': 'BlackScholes'},
                ],
                value='BlackScholes',
                style={'color': 'black'}
            ),
            html.Br(),
            
            # Professional Settings Accordion
            dbc.Accordion([
                dbc.AccordionItem([
                    html.Label("Optimization Method"),
                    dcc.Dropdown(
                        id="calib-method",
                        options=[
                            {'label': 'L-BFGS-B (Bounded)', 'value': 'L-BFGS-B'},
                            {'label': 'SLSQP (Bounded)', 'value': 'SLSQP'},
                            {'label': 'TNC (Bounded)', 'value': 'TNC'},
                            {'label': 'Nelder-Mead (Unbounded)', 'value': 'Nelder-Mead'},
                            {'label': 'Powell (Bounded)', 'value': 'Powell'}
                        ],
                        value='L-BFGS-B',
                        style={'color': 'black', 'font-size': '0.9rem'}
                    ),
                    html.Hr(),
                    html.Label("Pricing Method"),
                    dcc.Dropdown(
                        id="pricing-method",
                        options=[
                            {'label': 'Lewis (Integration)', 'value': 'Lewis'},
                            {'label': 'Carr-Madan (FFT)', 'value': 'Carr-Madan'}
                        ],
                        value='Lewis',
                        style={'color': 'black', 'font-size': '0.9rem'}
                    ),
                    html.Hr(),
                    html.Label("Initial Guesses"),
                    
                    # Dynamic Initial Guess Inputs (Heston Default)
                    html.Div(id="initial-guess-container", children=[
                         # Will be populated by callback based on model
                         dbc.InputGroup([
                             dbc.InputGroupText("v0"), dbc.Input(id="guess-p0", value=0.04, type="number", step=0.01)
                         ], size="sm", className="mb-1"),
                         dbc.InputGroup([
                             dbc.InputGroupText("κ"), dbc.Input(id="guess-p1", value=1.5, type="number", step=0.1)
                         ], size="sm", className="mb-1"),
                         dbc.InputGroup([
                             dbc.InputGroupText("θ"), dbc.Input(id="guess-p2", value=0.04, type="number", step=0.01)
                         ], size="sm", className="mb-1"),
                         dbc.InputGroup([
                             dbc.InputGroupText("ξ"), dbc.Input(id="guess-p3", value=0.3, type="number", step=0.1)
                         ], size="sm", className="mb-1"),
                         dbc.InputGroup([
                             dbc.InputGroupText("ρ"), dbc.Input(id="guess-p4", value=-0.5, type="number", step=0.1)
                         ], size="sm", className="mb-1"),
                         dcc.Input(id="guess-p5", value=0.1, type="hidden"),
                         dcc.Input(id="guess-p6", value=-0.1, type="hidden"),
                         dcc.Input(id="guess-p7", value=0.1, type="hidden"),
                    ])
                ], title="Advanced Settings")
            ], start_collapsed=True, className="mb-3"),
            
            dbc.Button("Calibrate Model", id="calibrate-btn", color="warning", className="w-100", size="lg"),
            
            # Stores
            dcc.Store(id="market-data-store"),
            dcc.Store(id="calibration-result-store"),
            
        ])
    ], className="h-100", style=CARD_STYLE)

    # Middle Panel: Visualizations
    middle_panel = html.Div([
        # Calibration Surface 
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Calibration Surface (Market vs Model)", style=HEADER_STYLE),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-surface",
                            type="circle",
                            color="#00ffff",  # Cyan glow
                            children=dcc.Graph(id="calibration-surface", style={"height": "600px"})
                        )
                    ])
                ], style=CARD_STYLE)
            ])
        ]),
        html.Br(),
        
        # Calibration Error & Smile
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Calibration Error (Residuals)", style=HEADER_STYLE),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-error",
                            type="circle",
                            color="#00ffff",
                            children=dcc.Graph(id="calibration-error", style={"height": "400px"})
                        )
                    ])
                ], style=CARD_STYLE)
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Implied Volatility Smile", style=HEADER_STYLE),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-smile",
                            type="circle",
                            color="#00ffff",
                            children=dcc.Graph(id="calibration-iv-smile", style={"height": "400px"})
                        )
                    ])
                ], style=CARD_STYLE)
            ], width=6),
        ]),
        html.Br(),

        # Bottom Panel: Equations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Important Equations", style=HEADER_STYLE),
                    dbc.CardBody([
                        html.Div(id="equations-container")
                    ])
                ], style=CARD_STYLE)
            ])
        ])
    ])

    # Right Panel: Reporting
    right_panel = html.Div([
        # Top Sub-panel: Price
        dbc.Card([
            dbc.CardHeader("Option Price", style=HEADER_STYLE),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-status",
                    type="dot",
                    color="#00ffff",
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.H6("Calibration Status", className="text-muted"),
                                html.Div(id="calibration-status", className="text-info")
                            ], width=12),
                        ]),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.H6("RMSE Error", className="text-muted"),
                                html.H4(id="calibration-rmse", className="text-danger")
                            ], width=12),
                        ])
                    ]
                )
            ])
        ], className="mb-3", style=CARD_STYLE),
        
        # Greeks Panel
        dbc.Card([
            dbc.CardHeader("Greeks & Metrics", style=HEADER_STYLE),
            dbc.CardBody([
                html.Div(id="calibrated-params-container", style={"color": "white"})
            ])
        ], className="mb-3", style=CARD_STYLE),

        # Parity Check
        dbc.Card([
             dbc.CardHeader("Put-Call Parity", style=HEADER_STYLE),
             dbc.CardBody([
                 html.Div(id="parity-check", style={"color": "white"})
             ])
        ], className="mb-3", style=CARD_STYLE),

        # Market Data Table
        dbc.Card([
             dbc.CardHeader("Generated Market Data", style=HEADER_STYLE),
             dbc.CardBody([
                 html.Div(id="market-data-table-container", style={"color": "white", "overflow-y": "scroll", "max-height": "400px"})
             ])
        ], style=CARD_STYLE)
        
    ], className="h-100")

    layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Option Price Calibration", className="text-center text-primary"), width=12)
        ], className="my-3"),
        
        dbc.Row([
            dbc.Col([
                html.H4("Configuration", className="text-center mb-3", style={"color": "#aaa"}),
                left_panel
            ], width=2),
            dbc.Col([
                html.H4("Visualization", className="text-center mb-3", style={"color": "#aaa"}),
                middle_panel
            ], width=8),
            dbc.Col([
                html.H4("Reporting", className="text-center mb-3", style={"color": "#aaa"}),
                right_panel
            ], width=2)
        ], className="g-3")
    ], fluid=True, style={"background-color": "#000000", "min-height": "100vh", "color": "white"})
    
    return layout

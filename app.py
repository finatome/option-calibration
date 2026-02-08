import dash
import dash_bootstrap_components as dbc
from src.dashboard.layout import create_layout
from src.dashboard.callbacks import register_callbacks

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="Discrete Derivative Pricing")

# Set the layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

# Expose server for WSGI
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8050)

from dash import Input, Output, State, html, dcc, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data.synthetic import generate_option_surface
from src.calibration.optimize import calibrate_model
from src.pricing.heston import heston_price
from src.pricing.merton import merton_jump_diffusion_price
from src.pricing.black_scholes import black_scholes_price

def bs_implied_vol(S0, K, T, r, price, option_type='C'):
    """
    Calculate Implied Volatility using Newton-Raphson.
    """
    sigma = 0.3 # Initial guess
    for i in range(100):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'C':
            price_curr = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            vega = S0 * norm.pdf(d1) * np.sqrt(T)
        else:
            price_curr = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            vega = S0 * norm.pdf(d1) * np.sqrt(T)
            
        diff = price - price_curr
        if abs(diff) < 1e-5:
            return sigma
        if abs(vega) < 1e-8:
            break
        sigma += diff / vega
    return sigma

def register_callbacks(app):
    
    # 1. Equations (Keep existing or simplified)
    @app.callback(
        Output("equations-container", "children"),
        [Input("pricing-model", "value")]
    )
    def update_equations(model):
        if model == 'BlackScholes':
            rows = [
                html.Tr([
                    html.Td("Asset Price SDE"), 
                    html.Td(dcc.Markdown(r"$$dS_t = r S_t dt + \sigma S_t dW_t$$", mathjax=True)), 
                    html.Td("Geometric Brownian Motion assumption.")
                ]),
                html.Tr([
                    html.Td("Call Price (BSM)"), 
                    html.Td(dcc.Markdown(r"$$C(S,K,T) = S_0 N(d_1) - K e^{-rT} N(d_2)$$", mathjax=True)), 
                    html.Td("Closed-form analytical solution.")
                ]),
                html.Tr([
                    html.Td("Calibration Objective"), 
                    html.Td(dcc.Markdown(r"$$\min_{\sigma} \sum_{i=1}^N (C_{mkt}^i - C_{BS}^i(\sigma))^2$$", mathjax=True)), 
                    html.Td("Minimize MSE between Market and Model prices.")
                ])
            ]
        elif model == 'Heston':
             rows = [
                html.Tr([
                    html.Td("Heston SDEs"), 
                    html.Td(dcc.Markdown(r"$$\begin{aligned} dS_t &= r S_t dt + \sqrt{v_t} S_t dW_t^S \\ dv_t &= \kappa(\theta - v_t)dt + \xi \sqrt{v_t} dW_t^v \end{aligned}$$", mathjax=True)), 
                    html.Td("Stochastic Volatility with Mean Reversion.")
                ]),
                html.Tr([
                    html.Td("Characteristic Func"), 
                    html.Td(dcc.Markdown(r"$$\phi(u) = \exp(Ak + B(u,t) + C(u,t)v_0)$$", mathjax=True)), 
                    html.Td("Fourier Transform of the log-price density.")
                ]),
                html.Tr([
                    html.Td("Pricing Integral"), 
                    html.Td(dcc.Markdown(r"$$C_{Heston} = S_0 - \frac{K e^{-rT}}{\pi} \int_{0}^{\infty} \text{Re}\left(\frac{e^{-iu k} \phi(u - i/2)}{u^2 + 1/4}\right) du$$", mathjax=True)), 
                    html.Td("Lewis (2001) approach via integration.")
                ]),
                html.Tr([
                    html.Td("Calibration Objective"), 
                    html.Td(dcc.Markdown(r"$$\min_{\Theta} \sum_{i=1}^N (C_{mkt}^i - C_{Heston}^i(\Theta))^2$$", mathjax=True)), 
                    html.Td(dcc.Markdown(r"Find $\Theta = \{v_0, \kappa, \theta, \xi, \rho\}$."))
                ])
            ]
        elif model == 'Merton':
             rows = [
                html.Tr([
                    html.Td("Jump Diffusion SDE"), 
                    html.Td(dcc.Markdown(r"$$\frac{dS_t}{S_{t-}} = (r-\lambda k)dt + \sigma dW_t + (e^J-1)dN_t$$", mathjax=True)), 
                    html.Td("GBM + Poisson Jump Process.")
                ]),
                html.Tr([
                    html.Td("Jump Parameters"), 
                    html.Td(dcc.Markdown(r"$$J \sim N(\mu_J, \delta^2), \quad k = E[e^J] - 1$$", mathjax=True)), 
                    html.Td("Log-normal jump size distribution.")
                ]),
                html.Tr([
                    html.Td("Pricing Series"), 
                    html.Td(dcc.Markdown(r"$$C_{Merton} = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T} (\lambda' T)^n}{n!} C_{BS}(S, K, T, r_n, \sigma_n)$$", mathjax=True)), 
                    html.Td("Infinite weighted sum of BS prices.")
                ]),
                html.Tr([
                    html.Td("Calibration Objective"), 
                    html.Td(dcc.Markdown(r"$$\min_{\Theta} \sum_{i=1}^N (C_{mkt}^i - C_{Merton}^i(\Theta))^2$$", mathjax=True)), 
                    html.Td(dcc.Markdown(r"Find $\Theta = \{\sigma, \lambda, \mu_J, \delta\}$."))
                ])
            ]
        return dbc.Table([
            html.Thead(html.Tr([
                html.Th("Component", style={"color": "white"}), 
                html.Th("Equation", style={"color": "white"}), 
                html.Th("Description", style={"color": "white"})
            ])),
            html.Tbody(rows)
        ], bordered=True, hover=True, striped=False, style={"background-color": "black", "color": "white", "font-size": "1.0rem"})

    # 2. Generate Synthetic Data
    @app.callback(
        [Output("market-data-store", "data"),
         Output("market-data-table-container", "children")],
        [Input("generate-btn", "n_clicks")],
        [State("synth-s0", "value"),
         State("synth-r", "value"),
         State("synth-kappa", "value"),
         State("synth-theta", "value"),
         State("synth-xi", "value"),
         State("synth-rho", "value"),
         State("synth-noise", "value")]
    )
    def generate_data(n_clicks, S0, r, kappa, theta, xi, rho, noise):
        if not n_clicks:
             return None, ""
             
        params = {
            "v0": 0.04, 
            "kappa": float(kappa), "theta": float(theta), "xi": float(xi), "rho": float(rho)
        }
        
        df = generate_option_surface(
            S0=float(S0), r=float(r), 
            min_strike=0.8, max_strike=1.2, num_strikes=8,
            min_expiry=0.2, max_expiry=2.0, num_expiries=5,
            model_name="Heston", model_params=params, 
            noise_level=float(noise) if noise else 0.0
        )
        
        table = dbc.Table.from_dataframe(df.round(4), striped=True, bordered=True, hover=True, size="sm", style={"color": "white"})
        return df.to_json(date_format='iso', orient='split'), table

    # 2.5 Dynamic Initial Guess Inputs
    @app.callback(
        Output("initial-guess-container", "children"),
        [Input("pricing-model", "value")]
    )
    def update_initial_guess_inputs(model):
        if model == 'Heston':
             return [
                 dbc.InputGroup([dbc.InputGroupText("v0"), dbc.Input(id="guess-p0", value=0.04, type="number", step=0.01)], size="sm", className="mb-1"),
                 dbc.InputGroup([dbc.InputGroupText("κ"), dbc.Input(id="guess-p1", value=1.5, type="number", step=0.1)], size="sm", className="mb-1"),
                 dbc.InputGroup([dbc.InputGroupText("θ"), dbc.Input(id="guess-p2", value=0.04, type="number", step=0.01)], size="sm", className="mb-1"),
                 dbc.InputGroup([dbc.InputGroupText("ξ"), dbc.Input(id="guess-p3", value=0.3, type="number", step=0.1)], size="sm", className="mb-1"),
                 dbc.InputGroup([dbc.InputGroupText("ρ"), dbc.Input(id="guess-p4", value=-0.5, type="number", step=0.1)], size="sm", className="mb-1"),
             ]
        elif model == 'Merton':
             return [
                 dbc.InputGroup([dbc.InputGroupText("σ"), dbc.Input(id="guess-p0", value=0.2, type="number", step=0.01)], size="sm", className="mb-1"),
                 dbc.InputGroup([dbc.InputGroupText("λ"), dbc.Input(id="guess-p1", value=1.0, type="number", step=0.1)], size="sm", className="mb-1"),
                 dbc.InputGroup([dbc.InputGroupText("μ_J"), dbc.Input(id="guess-p2", value=-0.1, type="number", step=0.05)], size="sm", className="mb-1"),
                 dbc.InputGroup([dbc.InputGroupText("σ_J"), dbc.Input(id="guess-p3", value=0.1, type="number", step=0.01)], size="sm", className="mb-1"),
                 dcc.Store(id="guess-p4", data=0), # Dummy for Heston compatibility
             ]
        elif model == 'BlackScholes':
             return [
                 dbc.InputGroup([dbc.InputGroupText("σ"), dbc.Input(id="guess-p0", value=0.2, type="number", step=0.01)], size="sm", className="mb-1"),
                 dcc.Store(id="guess-p1", data=0), dcc.Store(id="guess-p2", data=0),
                 dcc.Store(id="guess-p3", data=0), dcc.Store(id="guess-p4", data=0)
             ]
        return []

    # 3. Calibrate Model
    @app.callback(
        [Output("calibration-result-store", "data"),
         Output("calibration-status", "children"),
         Output("calibration-rmse", "children"),
         Output("calibrated-params-container", "children")],
        [Input("calibrate-btn", "n_clicks")],
        [State("pricing-model", "value"),
         State("market-data-store", "data"),
         State("calib-method", "value"),
         State("guess-p0", "value"), State("guess-p1", "value"),
         State("guess-p2", "value"), State("guess-p3", "value"),
         State("guess-p4", "value")]
    )
    def run_calibration(n_clicks, model_name, market_json, method, p0, p1, p2, p3, p4):
        if not n_clicks or not market_json:
            return None, "Waiting...", "---", ""
        
        df = pd.read_json(market_json, orient='split')
        
        # Collect Guesses
        initial_guess = []
        try:
            if model_name == 'Heston':
                initial_guess = [float(p0), float(p1), float(p2), float(p3), float(p4)]
            elif model_name == 'Merton':
                initial_guess = [float(p0), float(p1), float(p2), float(p3)]
            elif model_name == 'BlackScholes':
                initial_guess = [float(p0)]
        except:
             return None, html.Span("Invalid Initial Guess", className="text-danger"), "Error", ""

        result = calibrate_model(model_name, df, initial_guess=initial_guess, method=method)
        
        status_color = "text-success" if result['success'] else "text-warning"
        rmse = np.sqrt(result['error'])
        
        params_ui = []
        param_names = []
        if model_name == 'Heston':
            param_names = ['v0', 'kappa', 'theta', 'xi', 'rho']
        elif model_name == 'Merton':
            param_names = ['sigma', 'lambda', 'mu_j', 'sigma_j']
        elif model_name == 'BlackScholes':
            param_names = ['sigma']
            
        for i, val in enumerate(result['params']):
            name = param_names[i] if i < len(param_names) else f"p{i}"
            params_ui.append(dbc.Row([dbc.Col(html.Strong(f"{name}:"), width=6), dbc.Col(f"{val:.4f}", width=6)]))
            
        return result, html.Span(result['message'], className=status_color), f"{rmse:.5f}", params_ui

    # 4. Update Visualizations (Surface, Error, IV Smile)
    @app.callback(
        [Output("calibration-surface", "figure"),
         Output("calibration-error", "figure"),
         Output("calibration-iv-smile", "figure")],
        [Input("market-data-store", "data"),
         Input("calibration-result-store", "data")],
        [State("pricing-model", "value")]
    )
    def update_plots(market_json, calib_result, model_name):
        # Default empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark')
        
        if not market_json:
            return empty_fig, empty_fig, empty_fig
            
        df = pd.read_json(market_json, orient='split')
        
        # 1. Market Surface (Always show market data)
        fig_surface = go.Figure()
        fig_surface.add_trace(go.Scatter3d(
            x=df['Strike'], y=df['Maturity'], z=df['Price'],
            mode='markers', marker=dict(size=4, color='cyan', opacity=0.8),
            name='Market Data'
        ))
        
        # 2. Model Surface (If calibrated)
        df['Model_Price'] = np.nan
        if calib_result:
            optimized_params = calib_result['params']
            model_prices = []
            iv_market = []
            iv_model = []
            
            for index, row in df.iterrows():
                S0, K, T, r, opt_type = row['S0'], row['Strike'], row['Maturity'], row['r'], row['Type']
                
                # Pricing
                if model_name == 'Heston':
                    p = heston_price(S0, K, T, r, *optimized_params, opt_type)
                elif model_name == 'Merton':
                    p = merton_jump_diffusion_price(S0, K, T, r, *optimized_params, opt_type)
                elif model_name == 'BlackScholes':
                    p, _ = black_scholes_price(S0, K, T, r, optimized_params[0], opt_type)
                else:
                    p = 0.0
                model_prices.append(p)
                
                # IV Calculation (for Smile)
                # Only calculate for one maturity to keep plot clean, or plot all?
                # Let's calculate for all and filter later
                iv_mkt = bs_implied_vol(S0, K, T, r, row['Price'], opt_type)
                iv_mod = bs_implied_vol(S0, K, T, r, p, opt_type)
                iv_market.append(iv_mkt)
                iv_model.append(iv_mod)

            df['Model_Price'] = model_prices
            df['IV_Market'] = iv_market
            df['IV_Model'] = iv_model
            df['Error'] = df['Price'] - df['Model_Price']
            
            # Add Model Surface
            fig_surface.add_trace(go.Mesh3d(
                x=df['Strike'], y=df['Maturity'], z=df['Model_Price'],
                color='orange', opacity=0.5, name='Calibrated Model'
            ))
            
            # 3. Error Plot
            fig_error = go.Figure()
            fig_error.add_trace(go.Scatter3d(
                x=df['Strike'], y=df['Maturity'], z=df['Error'],
                mode='markers', marker=dict(size=5, color=df['Error'], colorscale='RdBu', showscale=True),
                name='Residuals'
            ))
            fig_error.update_layout(title="Calibration Residuals", scene=dict(zaxis_title='Error'), template='plotly_dark')
            
            # 4. IV Smile (Select one maturity)
            unique_expiries = sorted(df['Maturity'].unique())
            target_T = unique_expiries[len(unique_expiries)//2] # Middle maturity
            subset = df[df['Maturity'] == target_T].sort_values('Strike')
            
            fig_smile = go.Figure()
            fig_smile.add_trace(go.Scatter(x=subset['Strike'], y=subset['IV_Market'], mode='markers', name=f'Market IV (T={target_T:.2f})'))
            fig_smile.add_trace(go.Scatter(x=subset['Strike'], y=subset['IV_Model'], mode='lines', name=f'Model IV (T={target_T:.2f})'))
            fig_smile.update_layout(title=f"Implied Volatility Smile (T={target_T:.2f})", xaxis_title="Strike", yaxis_title="Implied Vol", template='plotly_dark')
            
        else:
            fig_error = empty_fig
            fig_smile = empty_fig
            fig_surface.update_layout(title="Market Data Only (Calibrate to see Model)")

        fig_surface.update_layout(title="Option Price Surface", scene=dict(xaxis_title='Strike', yaxis_title='Maturity', zaxis_title='Price'), template='plotly_dark')

        return fig_surface, fig_error, fig_smile

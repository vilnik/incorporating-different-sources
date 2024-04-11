from itertools import product


def get_color_from_display_name(display_name):
    # Quantstats ['#FFD700', '#E63946', '#A8DADC', '#457B9D', '#FF69B4', '#1D3557', '#F4A261', '#2A9D8F', '#9370DB', '#9DC209']
    colors = {"S&P 500": "#FFD700",
              "VW": "#E63946",
              "EW": "#A8DADC",
              "Conjugate HF-VIX VW": "#457B9D",
              "Conjugate HF-VIX EW": "#4D85A6",
              "Conjugate HF-EPU VW": "#FF69B4",
              "Conjugate HF-EPU EW": "#FF7F50",
              "Jeffreys": "#1D3557",
              "Shrinkage": "#F4A261",
              "Jorion Hyperpar.": "#2A9D8F",
              "Black-Litterman": "#9370DB",
              "Greyserman Hiera.": "#9DC209"}

    return colors[display_name]


def get_display_name_from_full_name(full_name):

    if "conjugate_hf_vix_vw" in full_name:
        display_name = "Conjugate HF-VIX VW"
    elif "conjugate_hf_vix_ew" in full_name:
        display_name = "Conjugate HF-VIX EW"
    elif "conjugate_hf_epu_vw" in full_name:
        display_name = "Conjugate HF-EPU VW"
    elif "conjugate_hf_epu_ew" in full_name:
        display_name = "Conjugate HF-EPU EW"
    elif "jeffreys" in full_name:
        display_name = "Jeffreys"
    elif "black_litterman" in full_name:
        display_name = "Black-Litterman"
    elif "shrinkage" in full_name:
        display_name = "Shrinkage"
    elif "jorion" in full_name:
        display_name = "Jorion Hyperpar."
    elif "greyserman" in full_name:
        display_name = "Greyserman Hiera."
    elif "vw" in full_name:
        display_name = "VW"
    elif "ew" in full_name:
        display_name = "EW"
    else:
        display_name = None

    return display_name

def create_portfolio_specs():
    weighting_strategies = ["vw", "ew", "conjugate_hf_vix_vw", "conjugate_hf_epu_vw", "jeffreys",  "shrinkage", "jorion", "black_litterman", "greyserman"]
    #weighting_strategies = ["conjugate_hf_vix_vw", "conjugate_hf_epu_vw"]
    sizes = [50]
    risk_aversions = [5]
    #turnover_costs = [0, 5, 10, 15, 20, 25, 30, 35]
    turnover_costs = [15]
    rebalancing_frequencies = ["monthly"]
    rolling_window = [250]
    rolling_window_frequenies = ["weekly"]
    #mcm_scalings = [0.001, 1, 5, 20]
    mcm_scalings = [1]

    all_portfolio_specs = {}

    for weighting_strategy in weighting_strategies:
        # Determine valid risk aversions based on the weight spec
        valid_risk_aversions = [None] if weighting_strategy in {"vw", "ew"} else risk_aversions
        valid_mcm_scalings = [None] if not weighting_strategy in {"conjugate_hf_vix_vw", "conjugate_hf_vix_ew", "conjugate_hf_epu_vw", "conjugate_hf_epu_ew"} else mcm_scalings

        for size, risk, turnover, freq, window, window_freq, mcm_scaling in product(
            sizes, valid_risk_aversions, turnover_costs, rebalancing_frequencies, rolling_window, rolling_window_frequenies, valid_mcm_scalings):

            risk_label = "NA" if risk is None else risk
            mcm_scaling_label = "NA" if mcm_scaling is None else mcm_scaling

            key = f"weighting_strategy_{weighting_strategy}_size_{size}_risk_aversion_{risk_label}_turnover_cost_{turnover}_rebalancing_frequency_{freq}_rolling_window_{window}_rolling_window_frequency_{window_freq}_mcm_scaling_{mcm_scaling_label}"
            display_name = get_display_name_from_full_name(key)

            all_portfolio_specs[key] = {
                "weighting_strategy": weighting_strategy,
                "size": size,
                "risk_aversion": risk,
                "turnover_cost": turnover,
                "rebalancing_frequency": freq,
                "rolling_window": window,
                "rolling_window_frequency": window_freq,
                "mcm_scaling": mcm_scaling,
                "display_name": display_name
            }

    return all_portfolio_specs

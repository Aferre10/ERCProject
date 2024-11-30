import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from fpdf import FPDF
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import plotly.express as px
import matplotlib.pyplot as plt
from ew_calculations import (
    load_external_data,
    process_equal_weighted_portfolios,
    compute_performance_metrics,
    generate_asset_class_composition,
    file_paths,
)
from erc_calculations import (
    load_external_data2,
    process_equal_weighted_portfolios,
    compute_performance_metrics,
    generate_asset_class_composition,
    compute_class_correlation_matrix,
    generate_fees_table,
    generate_erc_portfolios,
    generate_risk_contribution_table,
    calculate_yearly_and_final_metrics_with_fees,
    analyze_and_plot_portfolios,
    calculate_daily_returns_with_fees,
    generate_erc_portfolio,
    calculate_yearly_and_final_metrics_with_fees_for_portfolio,
    plot_cumulative_and_drawdown,
    align_daily_returns,
    plot_asset_allocation_over_time_for_portfolio,
    display_asset_allocation_over_time_for_portfolio,
    extract_weights_over_time,
    plot_risk_contribution_over_time_for_portfolio,
    display_risk_contribution_over_time_for_portfolio,
    extract_risk_contributions_over_time,
    select_and_display_portfolio_performance_with_period,
    plot_returns_comparison,
    plot_interactive_cumulative_and_drawdowns
)


# ----------------------------- Streamlit Configuration -----------------------------
st.set_page_config(
    page_title="Alpine Asset Management - ERC Portfolio Analysis", layout="wide"
)

# ======== Custom CSS for Styling ========
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Times:wght@400;700&display=swap');
    body {
        background-color: #E6F0FA;
        font-family: 'Times New Roman', sans-serif;
        color: #002366;
    }
    .main-header {
        background-color: #00509E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: #ffffff;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .image-banner {
        background-image: url('https://images.pexels.com/photos/753772/pexels-photo-753772.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');
        background-size: cover;
        background-position: center;
        height: 300px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .nav-buttons button {
        width: 100%;
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        background-color: #00509E;
        border: none;
        border-radius: 5px;
        margin: 5px;
        cursor: pointer;
    }
    .nav-buttons button:hover {
        background-color: #003366;
    }
    .banner {
        display: flex;
        align-items: center;
        background-color: #003366; /* Dark blue background */
        padding: 10px 20px;
        border-radius: 8px; /* Rounded corners */
        margin-bottom: 20px; /* Space below the banner */
    }
    .logo {
        border-radius: 50%; /* Makes the logo round */
        width: 80px; /* Logo size */
        margin-right: 20px; /* Space between logo and text */
    }
    .text {
        font-size: 32px; /* Larger font size */
        font-weight: normal; /* Not bold */
        color: #ffffff; /* White text color */
        font-family: 'Times', sans-serif; /* Rounded font */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======== Main Header ========
# Online image URL
image_path = "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Logo.jpeg"

try:
    # Fetch the image from the URL
    response = requests.get(image_path)
    response.raise_for_status()  # Check if the request was successful

    # Convert the image content to base64 for embedding in HTML
    encoded_image = base64.b64encode(response.content).decode()

    # Show logo and text in Streamlit with styled HTML embedding
    st.markdown(
        f"""
        <div class="banner">
            <img src="data:image/jpeg;base64,{encoded_image}" alt="Logo" class="logo">
            <div class="text">Alpine Asset Management</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

except requests.exceptions.RequestException as e:
    # Handle errors if the image cannot be fetched
    st.error(f"Unable to load the logo from the URL. Error: {e}")

# Display image banner
st.markdown('<div class="image-banner"></div>', unsafe_allow_html=True)


# ======== Load Data ========
# ----------------------------- Paths to Data -----------------------------
rf_path = "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/NEW_RF_ERC.xlsx"
benchmarks_path = "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/NEW_BMK_ERC.xlsx"


@st.cache_data
def load_data():
    """Load and preprocess all data."""
    # Load external data (risk-free rate and benchmarks)
    RF, Benchs = load_external_data()
    risk_free_rate_series = RF['RF6M'] / 252  # Convert annualized rate to daily
    market_returns = (Benchs["AQR MULTI-ASSET FUND I"] / Benchs["AQR MULTI-ASSET FUND I"].shift(1) - 1).dropna()

    # Define file paths for asset classes
    file_paths = {
        "Equities": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Equities.xlsx",
        "Fixed Income": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_FixedIncome.xlsx",
        "Real Estate": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_RealEstate.xlsx",
        "Commodities": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Commodities.xlsx",
        "Alternative Investments": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_AlternativeInvestments.xlsx",
        "Money Market": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_MoneyMarket.xlsx",
        "Currencies": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Currencies.xlsx"
    }

    # Process portfolios
    equal_weighted_portfolios = process_equal_weighted_portfolios(file_paths)

    # Compute performance metrics
    performance_df = compute_performance_metrics(equal_weighted_portfolios, risk_free_rate_series, market_returns)

    # Generate asset class composition
    assets_table = generate_asset_class_composition(file_paths)

    return equal_weighted_portfolios, performance_df, assets_table

# Load all data
equal_weighted_portfolios, performance_df, assets_table = load_data()

# ----------------------------- Load Data -----------------------------

rebalancing_freq_months = 6
# Dynamically load data based on rebalancing frequency
@st.cache_data
def load_all_data2(rebalancing_freq_months=6):  # Default to 6 months
    RF, Benchs = load_external_data2()
    risk_free_rate_series = RF[f'RF{rebalancing_freq_months}M'] / 100 / 252
    market_prices = Benchs["AQR MULTI-ASSET FUND I"]
    market_returns = market_prices.pct_change().dropna()
    file_paths = {
        "Equities": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Equities.xlsx",
        "Fixed Income": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_FixedIncome.xlsx",
        "Real Estate": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_RealEstate.xlsx",
        "Commodities": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Commodities.xlsx",
        "Alternative Investments": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_AlternativeInvestments.xlsx",
        "Money Market": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_MoneyMarket.xlsx",
        "Currencies": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Currencies.xlsx"
    }
    equal_weighted_portfolios = process_equal_weighted_portfolios(file_paths)
    return RF, Benchs, risk_free_rate_series, market_returns, file_paths, equal_weighted_portfolios



# Load data with the selected rebalancing frequency
RF, Benchs, risk_free_rate_series, market_returns, file_paths, equal_weighted_portfolios = load_all_data2(rebalancing_freq_months)

# ======== Navigation Buttons with State Management ========
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "About Us"



# Horizontal layout for buttons
cols = st.columns([1.5, 2.5, 1.5, 2.5, 1.5, 2, 1, 1])  # Adjust the number of columns to match your buttons

with cols[0]:
    if st.button("About Us", key="about"):
        st.session_state.selected_page = "About Us"
with cols[1]:
    if st.button("Portfolio Characteristics", key="portfolio"):
        st.session_state.selected_page = "Portfolio Characteristics"
with cols[2]:
    if st.button("Client Profile", key="client"):
        st.session_state.selected_page = "Client Profile"
with cols[3]:
    if st.button("Portfolio Performance", key="performance"):
        st.session_state.selected_page = "Portfolio Performance"
with cols[4]:
    if st.button("Managers", key="managers"):
        st.session_state.selected_page = "Managers"
with cols[5]:
    if st.button("Fees and Minimums", key="fees"):
        st.session_state.selected_page = "Fees and Minimums"
with cols[6]:
    if st.button("News", key="news"):
        st.session_state.selected_page = "News"
with cols[7]:
    if st.button("Contact Us", key="Contact US"):
        st.session_state.selected_page = "Contact Us"


selected_page = st.session_state.selected_page

def about_us():
    # ======== About the Fund ========
    st.markdown(
        """
        <div style="
            background-color: #ADD8E6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            width: 100%;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h2 style="color: #00008B; font-size: 32px;">Welcome to Alpine Asset Management</h2>
            <i style="color: #000;">Your trusted partner in financial growth and security.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Add spacing
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

    # Quick Stats Section
    st.markdown(
        """
        <div style="
            background-color: #003366;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 24px;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 30px;
        ">
            Quick Infos
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            '<div style="background-color: #003366; padding: 20px; border-radius: 10px; text-align: center;">'
            '<div style="font-size: 18px; color: white;"><strong>DIVERSIFIED ALLOCATION</strong><br>'
            '<span style="font-size: 22px;">7 Asset Classes</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div style="background-color: #003366; padding: 20px; border-radius: 10px; text-align: center;">'
            '<div style="font-size: 18px; color: white;"><strong>PORTFOLIO OPTIMIZATION</strong><br>'
            '<span style="font-size: 22px;">Risk Allocation</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div style="background-color: #003366; padding: 20px; border-radius: 10px; text-align: center;">'
            '<div style="font-size: 18px; color: white;"><strong>ALLOCATION CHOICE</strong><br>'
            '<span style="font-size: 22px;">Tailored Strategy</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div style="background-color: #003366; padding: 20px; border-radius: 10px; text-align: center;">'
            '<div style="font-size: 18px; color: white;"><strong>INCEPTION DATE</strong><br>'
            '<span style="font-size: 22px;">30/11/2024</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            '<div style="background-color: #003366; padding: 20px; border-radius: 10px; text-align: center;">'
            '<div style="font-size: 18px; color: white;"><strong>SWISS BASED</strong><br>'
            '<span style="font-size: 22px;">FINMA Regulated</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

 # Add spacing
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # Two-column layout: text on the left, image on the right
    col1, col2 = st.columns([3, 2])  # Adjust column widths as needed

    with col1:
        # Presentation Text Section
        st.markdown(
            """
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h3 style="color: #00008B; text-align: center;">About Us</h3>
                <p style="font-size: 18px; line-height: 1.6; text-align: justify; color: #000;">
                    Alpine Asset Management is a Swiss-based investment firm specializing in tailored financial solutions for high-net-worth clients and institutional investors. Our mission is to provide robust, risk-conscious strategies that prioritize long-term stability and sustainable growth. With a focus on serving clients such as pension funds—entrusted with safeguarding retirees’ futures—we emphasize investment approaches that minimize unnecessary risks while delivering consistent returns.
                </p>
                <h4 style="color: #00008B;">Our Core Offering</h4>
                <p style="font-size: 16px; line-height: 1.6; text-align: justify; color: #000;">
                    Our flagship offering is the Equally Weighted Risk Contribution (ERC) portfolio, a sophisticated strategy designed to achieve balanced risk distribution across asset classes. By allocating risk equally, we ensure no single asset class disproportionately impacts portfolio performance, providing greater stability in volatile markets.
                </p>
                <h4 style="color: #00008B;">Our Asset Classes</h4>
                <ul style="font-size: 16px; line-height: 1.6; color: #000;">
                    <li><strong>Equities:</strong> Capture long-term growth opportunities in global markets.</li>
                    <li><strong>Commodities:</strong> Hedge against inflation and market volatility.</li>
                    <li><strong>Currencies:</strong> Diversify exposure with foreign exchange investments.</li>
                    <li><strong>Money Market:</strong> Secure short-term liquidity with low-risk instruments.</li>
                    <li><strong>Alternative Investments:</strong> Explore innovative opportunities in hedge funds, private equity, and more.</li>
                    <li><strong>Real Estate:</strong> Achieve consistent income and capital appreciation through global property markets.</li>
                    <li><strong>Fixed Income:</strong> Preserve capital and generate steady returns via bonds and other debt securities.</li>
                </ul>
                <h4 style="color: #00008B;">Our Philosophy</h4>
                <p style="font-size: 16px; line-height: 1.6; text-align: justify; color: #000;">
                    At Alpine Asset Management, we believe that effective diversification is the cornerstone of sound investment. By designing globally diversified portfolios, we help our clients navigate complex market environments while remaining resilient against economic shocks. Our ERC methodology enhances this approach by maintaining balanced risk exposure, enabling clients to pursue their financial goals with confidence and precision.
                </p>
                <h4 style="color: #00008B;">Why Choose Us?</h4>
                <ul style="font-size: 16px; line-height: 1.6; color: #000;">
                    <li><strong>Tailored Investment Solutions:</strong> We cater specifically to high-net-worth individuals who value a personalized, strategic approach to investing.</li>
                    <li><strong>Balanced Risk Management:</strong> Our equally weighted methodology promotes stability and minimizes the risk of overexposure to any single asset class.</li>
                    <li><strong>Swiss Heritage and Expertise:</strong> Operating from one of the world's leading financial hubs, we combine Swiss precision, discretion, and trustworthiness to deliver exceptional service.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Display the image
        st.image(
            "https://images.pexels.com/photos/7243368/pexels-photo-7243368.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
            use_container_width=True,
        )

        # Add the caption in a styled box
        st.markdown(
            """
            <div style="
                background-color: #f8f9fa;
                padding: 10px;
                margin-top: 15px;
                border-radius: 10px;
                box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
                text-align: center;
                font-style: italic;
                font-size: 24px;
                line-height: 1.5;
                color: #00008B;
            ">
                Building a future of financial growth.
            </div>
            """,
            unsafe_allow_html=True,
        )




def portfolio_characteristics():
    st.markdown("## Portfolio Characteristics")

    # ---- Generate or Load Asset Classes ----
    assets_table = generate_asset_class_composition(file_paths)  # Ensure this function is defined elsewhere
    assets_in_classes = {col: assets_table[col].dropna().tolist() for col in assets_table.columns}

    # ---- Session State for Toggle Visibility ----
    if "show_correlation" not in st.session_state:
        st.session_state.show_correlation = False

    if "show_asset_composition" not in st.session_state:
        st.session_state.show_asset_composition = False

    if "show_performance_graph" not in st.session_state:
        st.session_state.show_performance_graph = False

    if "selected_class" not in st.session_state:
        st.session_state.selected_class = list(assets_in_classes.keys())[0]

    # ---- Button for Portfolio Performance Graph ----
    if st.button("Show Asset Classes Performance Graph"):
        st.session_state.show_performance_graph = not st.session_state.show_performance_graph

    if st.session_state.show_performance_graph:
        # Title for the section
        st.markdown('<h2 class="section-title">Asset Classes Performance</h2>', unsafe_allow_html=True)

        # Let the user select which portfolios to display
        selected_portfolios = st.multiselect(
            "Select Asset Classes to Display",
            options=equal_weighted_portfolios.keys(),
            default=list(equal_weighted_portfolios.keys())  # Show all by default
        )

        # Option to scale the graph (linear vs log)
        scale_type = st.radio("Select Scale Type", options=["Linear", "Logarithmic"], index=0)

        # Plotting the graph
        if selected_portfolios:
            # Set up the figure
            plt.figure(figsize=(8, 4))  # Smaller size (8 inches by 4 inches)

            for category in selected_portfolios:
                portfolio = equal_weighted_portfolios[category]
                cumulative_returns = (1 + portfolio["Daily Equal Weighted Return"]).cumprod()
                plt.plot(cumulative_returns, label=category, linewidth=1.5)  # Thinner lines for a cleaner look

            # Adjust scale type
            if scale_type == "Logarithmic":
                plt.yscale("log")

            # Customizing the plot appearance
            plt.title("Cumulative Returns of Equally Weighted Portfolios", fontsize=14, pad=10)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Cumulative Return", fontsize=12)
            plt.xticks(fontsize=10, rotation=30)  # Rotated x-axis labels for better readability
            plt.yticks(fontsize=10)
            plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Light grid
            plt.legend(fontsize=10, loc='upper left', frameon=False)  # Smaller legend

            # Render the plot in Streamlit
            st.pyplot(plt)
        else:
            st.warning("Please select at least one portfolio to display.")

    # ---- Performance Metrics ----
    st.markdown("### Performance Metrics")

    numeric_performance_df = performance_df.copy()
    numeric_performance_df["Category"] = list(equal_weighted_portfolios.keys())

    for column in numeric_performance_df.columns:
        if column != "Category":
            numeric_performance_df[column] = pd.to_numeric(numeric_performance_df[column], errors="coerce")

    numeric_performance_df.set_index("Category", inplace=True)
    styled_table = numeric_performance_df.style.background_gradient(cmap="Blues").format("{:.4f}")
    st.dataframe(styled_table, use_container_width=True)

    asset_classes = list(assets_in_classes.keys())
    benefits = [
        "Global growth opportunities",
        "Stable income and capital preservation",
        "Income and inflation hedge",
        "Diversification and inflation hedge",
        "Access to innovative strategies",
        "Liquidity with minimal risk",
        "Hedge and exposure to global currencies",
    ]

    # Create a Plotly bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=asset_classes,
                y=[1] * len(asset_classes),  # Dummy values to create equal height
                text=benefits,
                textposition="inside",
                insidetextanchor="middle",
                marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]),
            )
        ]
    )

    fig.update_layout(
        title="Key Benefits of Asset Classes",
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        xaxis=dict(showticklabels=True, tickangle=-45),
        yaxis=dict(visible=False),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Button for Asset Class Details ----
    if st.button("Asset Class Composition"):
        st.session_state.show_asset_composition = not st.session_state.show_asset_composition

    if st.session_state.show_asset_composition:
        selected_class = st.selectbox(
            "Select an Asset Class:",
            list(assets_in_classes.keys()),
            index=list(assets_in_classes.keys()).index(st.session_state.selected_class),
        )
        st.session_state.selected_class = selected_class
        assets = assets_in_classes[selected_class]

        predefined_summaries = {
            "Equities": "Explore growth opportunities worldwide. Investing in global equities provides diversified exposure to international markets, spanning developed and emerging economies. This asset class offers broad sector representation and geographic diversification, helping reduce region-specific risks while capturing the benefits of global economic trends.",
            "Fixed Income": "Preserve capital and earn steady returns. Fixed Income investments include government and corporate bonds, offering stability and predictable income. These assets are ideal for risk-averse investors seeking protection against market volatility.",
            "Real Estate": "Achieve steady income and long-term growth. Real Estate investments include global property markets, providing consistent cash flow and a hedge against inflation.",
            "Commodities": "Hedge against inflation and market volatility. Commodities investments include gold, silver, and agricultural products. They offer diversification and protection against economic uncertainties.",
            "Alternative Investments": "Explore non-traditional growth opportunities. Alternative investments span private equity, hedge funds, and other innovative strategies. These assets provide portfolio diversification and access to specialized markets.",
            "Money Market": "Secure short-term liquidity with minimal risk. Money Market instruments focus on Treasury bills and high-quality debt securities. These assets are perfect for preserving capital while maintaining liquidity.",
            "Currencies": "Diversify with global currency exposure. Currency investments involve major and emerging market currencies. They enhance portfolio diversification and hedge against foreign exchange risk.",
        }

        st.markdown(f"### {selected_class} - Asset Class Details")
        st.markdown(predefined_summaries.get(selected_class, "No summary available for this class."))

        st.markdown("#### Assets in this Class:")
        st.table(pd.DataFrame({"Assets": assets}))

    # ---- Button for Correlation Matrix ----
    if st.button("Correlation Matrix"):
        st.session_state.show_correlation = not st.session_state.show_correlation

    if st.session_state.show_correlation:
        st.markdown("### Asset Classes Correlation ")

        combined_returns = pd.concat(
            [portfolio["Daily Equal Weighted Return"] for portfolio in equal_weighted_portfolios.values()],
            axis=1,
        )
        combined_returns.columns = list(equal_weighted_portfolios.keys())
        correlation_matrix = combined_returns.corr()

        styled_corr = correlation_matrix.style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}")
        st.dataframe(styled_corr, use_container_width=True)
        st.markdown(
            """
            ### Why is the Correlation Matrix Important?

            1. **Understanding Diversification**:
                - A correlation matrix shows how different asset classes move relative to each other. Values close to **1** mean they move in the same direction, while values close to **-1** mean they move in opposite directions.
                - Diversification works best when assets have low or negative correlations, as this reduces the risk of losses across your portfolio.

            2. **Risk Reduction**:
                - Investing in asset classes with low correlations helps smooth out the portfolio's overall performance. For example, if one asset class is underperforming, another with low correlation may be performing better, reducing the overall impact on your investment.

            3. **Optimal Asset Allocation**:
                - The correlation matrix helps us decide how to distribute investments across different asset classes. It ensures a balanced approach by identifying combinations of assets that complement each other, rather than amplifying risks by being too similar.

            ### What Does This Mean for You?
            - By analyzing correlations, we aim to build portfolios that are more resilient to market fluctuations. This approach helps achieve consistent returns while minimizing risks, tailored to your investment goals.
            """
        )




def process_and_round_image_from_url(image_url):
    """
    Fetches an image from a URL, applies a circular mask, and saves it temporarily for FPDF.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error if the fetch fails

        # Open the image and ensure it is in RGBA format
        img = Image.open(BytesIO(response.content)).convert("RGBA")

        # Create a circular mask
        width, height = img.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, width, height), fill=255)

        # Apply the circular mask to the image
        img.putalpha(mask)

        # Save the processed image to a temporary file
        rounded_image_path = "rounded_logo.png"  # Temporary file
        img.save(rounded_image_path, "PNG")
        return rounded_image_path
    except Exception as e:
        raise RuntimeError(f"Error processing image from URL: {e}")


def generate_pdf_report(client_data):
    # Online image URL
    image_url = "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Logo.jpeg"

    # Process the image to make it circular
    image_path = process_and_round_image_from_url(image_url)

    pdf = FPDF()
    pdf.add_page()

    # Add a blue banner at the top of the page
    banner_height = 40  # Height of the banner in mm
    pdf.set_fill_color(0, 0, 139)  # Dark blue (RGB)
    pdf.rect(0, 0, 210, banner_height, 'F')  # Draw the filled rectangle

    # Add the circular logo within the banner
    logo_size = 30  # Logo size in mm
    logo_x = 10  # Left margin
    logo_y = (banner_height - logo_size) / 2  # Center the logo vertically within the banner
    pdf.image(image_path, x=logo_x, y=logo_y, w=logo_size, h=logo_size)

    # Add the report title next to the logo within the banner
    pdf.set_font("Times", style="B", size=16)
    pdf.set_text_color(255, 255, 255)  # White text for the title
    pdf.set_xy(logo_x + logo_size + 10, banner_height / 2 - 5)  # Position title next to the logo
    pdf.cell(0, 10, txt="Client Investment Profile Report", ln=False, align='L')

    # Leave space below the banner for the first paragraph
    pdf.set_y(banner_height + 10)

    # Introductory Text
    pdf.set_font("Times", size=12)
    pdf.set_text_color(0, 0, 0)  # Black text for the body
    pdf.multi_cell(0, 10,
                   txt="Dear Client, based on your responses, we have identified the following investment opportunities "
                       "and recommended asset classes for your portfolio. Please review the recommended classes below "
                       "and let us know if you have any further questions or would like to adjust your preferences.")
    pdf.ln(10)

    # Client Data Titles in Bold (same line as answers)
    pdf.set_font("Times", style="B", size=12)

    # Client Name and its value on the same line
    pdf.cell(90, 10, txt="Client Name: ", ln=False)  # Title
    pdf.set_font("Times", size=12)
    pdf.cell(100, 10, txt=client_data['name'], ln=True)  # Answer

    # Investment Amount and its value on the same line
    pdf.set_font("Times", style="B", size=12)
    pdf.cell(90, 10, txt="Investment Amount: ", ln=False)  # Title
    pdf.set_font("Times", size=12)
    pdf.cell(100, 10, txt=f"${client_data['investment_amount']:,}", ln=True)  # Answer

    # Risk Tolerance and its value on the same line
    pdf.set_font("Times", style="B", size=12)
    pdf.cell(90, 10, txt="Risk Tolerance: ", ln=False)  # Title
    pdf.set_font("Times", size=12)
    pdf.cell(100, 10, txt=client_data['risk_tolerance'], ln=True)  # Answer

    # Time Horizon and its value on the same line
    pdf.set_font("Times", style="B", size=12)
    pdf.cell(90, 10, txt="Time Horizon: ", ln=False)  # Title
    pdf.set_font("Times", size=12)
    pdf.cell(100, 10, txt=client_data['time_horizon'], ln=True)  # Answer

    # Primary Objectives and its value on the same line
    pdf.set_font("Times", style="B", size=12)
    pdf.cell(90, 10, txt="Primary Objectives: ", ln=False)  # Title
    pdf.set_font("Times", size=12)
    pdf.cell(100, 10, txt=", ".join(client_data['objectives']), ln=True)  # Answer

    # Interests and its value on the same line
    pdf.set_font("Times", style="B", size=12)
    pdf.cell(90, 10, txt="Interests: ", ln=False)  # Title
    pdf.set_font("Times", size=12)
    pdf.cell(100, 10, txt=", ".join(client_data['interests']), ln=True)  # Answer

    # Rebalancing Frequency and its value on the same line
    pdf.set_font("Times", style="B", size=12)
    pdf.cell(90, 10, txt="Rebalancing Frequency: ", ln=False)  # Title
    pdf.set_font("Times", size=12)
    pdf.cell(100, 10, txt=client_data['rebalance_frequency'], ln=True)  # Answer
    pdf.ln(10)

    # Recommended Asset Classes
    pdf.set_font("Times", style="B", size=14)
    pdf.cell(200, 10, txt="Recommended Asset Classes:", ln=True)
    pdf.ln(5)

    predefined_summaries = {
        "Equities": "Equities offer substantial growth potential by investing in companies poised for long-term success. Your investment in equities aims to capture these growth opportunities while balancing the inherent market volatility. The strategy focuses on selecting companies with strong fundamentals and long-term value, ensuring your portfolio benefits from global economic trends while mitigating short-term market swings.",

        "Fixed Income": "Fixed Income investments provide a cornerstone of stability within your portfolio, delivering predictable returns and safeguarding capital. These assets are carefully selected to minimize risk while ensuring steady income streams. Your investment in this asset class aims to preserve wealth over the long term, generating reliable returns that complement more dynamic aspects of your portfolio, without unnecessary exposure to market turbulence.",

        "Real Estate": "Real Estate investments offer consistent and secure returns through income-generating properties and appreciation in value. This asset class plays a key role in your strategy, serving both as a hedge against inflation and as a method for diversifying your holdings. With an emphasis on high-quality, low-risk properties, your real estate investments are positioned to enhance stability and long-term growth without undue market fluctuations.",

        "Commodities": "Commodities are a vital component of a diversified portfolio, providing exposure to essential resources such as precious metals, oil, and agricultural products. These investments help guard against inflation and offer a layer of protection from economic instability. Through careful selection, your exposure to commodities helps balance your portfolio by integrating global trends and providing stability, ensuring resilience even in uncertain times.",

        "Alternative Investments": "Alternative Investments bring unique opportunities that are not tied to traditional asset classes, offering the potential for uncorrelated returns. Your strategy focuses on low-risk, high-quality alternatives such as private equity and hedge funds, chosen for their ability to complement more traditional assets while providing diversification. These investments are carefully vetted to ensure that they align with your goal of steady, long-term growth with minimal exposure to traditional market risks.",

        "Money Market": "Money Market instruments, including short-term government securities and highly liquid assets, are essential for maintaining stability and ensuring liquidity within your portfolio. These investments offer a safe harbor for your wealth, providing a low-risk, accessible option to preserve capital while earning modest returns. Their role in your strategy is to provide a secure, short-term investment avenue with minimal exposure to market volatility.",

        "Currencies": "Currency investments offer an effective means of diversifying your portfolio while protecting it from fluctuations in global exchange rates. Your investments can provide stability and risk mitigation, particularly in international markets. This strategy helps safeguard your wealth from currency devaluation and adds an additional layer of protection in an ever-changing global economic landscape."
    }

    # Loop through the recommended classes and display the asset summaries
    for cls in client_data["recommended_classes"]:
        pdf.set_font("Times", style="B", size=12)
        pdf.cell(200, 10, txt=f"- {cls}", ln=True)
        pdf.set_font("Times", size=12)
        pdf.multi_cell(0, 10, txt=predefined_summaries[cls])
        pdf.ln(5)

    # Signature Section
    pdf.ln(20)
    pdf.set_font("Times", style="B", size=12)
    pdf.cell(200, 10, txt="Signature Section", ln=True)
    pdf.ln(5)
    pdf.set_font("Times", size=12)
    pdf.cell(200, 10, txt="Client Signature: ________________________________", ln=True)
    pdf.cell(200, 10, txt="Alpine Asset Management Representative: ________________________________", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Alpine Asset Management", align='C')

    return pdf.output(dest="S").encode("latin1")





def client_profile():
    st.markdown('<h2 class="section-title">Client Profile</h2>', unsafe_allow_html=True)

    # ---- Step 1: Client Basic Information ----
    client_name = st.text_input("Enter your name:")
    investment_amount = st.number_input(
        "Enter your investment amount (Minimum CHF5,000,000):",
        min_value=5000000,
        step=100000,
        value=5000000,
    )

    # ---- Step 2: Known Asset Classes ----
    st.markdown("### Do you already know the asset classes you want to invest in?")
    known_classes = st.multiselect(
        "If yes, select them here:",
        ["Equities", "Fixed Income", "Real Estate", "Commodities", "Alternative Investments", "Money Market", "Currencies"],
    )

    # ---- Step 3: Investment Objectives ----
    st.markdown("### What are your primary investment objectives?")
    objectives = st.multiselect(
        "Select all that apply:",
        [
            "Increase long-term capital",
            "Generate stable income",
            "Protect against inflation",
            "Diversify my portfolio",
        ],
    )

    # ---- Step 4: Risk Tolerance ----
    st.markdown("### What level of risk are you comfortable with?")
    risk_tolerance = st.radio(
        "Select one:",
        ["Low Risk", "Moderate Risk", "High Risk"],
    )

    # ---- Step 5: Time Horizon ----
    st.markdown("### What is your investment time horizon?")
    time_horizon = st.radio(
        "Select one:",
        ["Less than 3 years", "3 to 7 years", "More than 7 years"],
    )

    # ---- Step 6: Rebalancing Frequency ----
    st.markdown("### How frequently would you like your portfolio to be rebalanced?")
    rebalance_frequency = st.radio(
        "Select one:",
        [ "Quarterly", "Semi-Annually", "Annually",],
    )

    # ---- Step 7: Interests ----
    st.markdown("### What themes or domains interest you the most?")
    interests = st.multiselect(
        "Select all that apply:",
        [
            "Technologies and growth",
            "Property development and long-term investments",
            "Stable financial instruments like bonds",
            "International trade (currencies or commodities)",
        ],
    )

    # ---- Generate Recommendations ----
    if st.button("Generate Recommendations"):
        recommended_classes = []

        # Logic for recommendations
        if "Increase long-term capital" in objectives or "Technologies and growth" in interests:
            recommended_classes.append("Equities")
        if "Generate stable income" in objectives or "Stable financial instruments like bonds" in interests:
            recommended_classes.append("Fixed Income")
        if "Protect against inflation" in objectives or "Property development and long-term investments" in interests:
            recommended_classes.append("Real Estate")
        if "Diversify my portfolio" in objectives or "International trade (currencies or commodities)" in interests:
            recommended_classes.extend(["Commodities", "Currencies"])
        if "High Risk" in risk_tolerance:
            recommended_classes.append("Alternative Investments")
        if "Less than 3 years" in time_horizon:
            recommended_classes.append("Money Market")

        # Add known classes and ensure uniqueness
        recommended_classes = list(set(recommended_classes + known_classes))

        # Automatically add missing classes if fewer than 3
        all_classes = ["Equities", "Fixed Income", "Real Estate", "Commodities", "Alternative Investments", "Money Market", "Currencies"]
        while len(recommended_classes) < 3:
            for cls in all_classes:
                if cls not in recommended_classes:
                    recommended_classes.append(cls)
                if len(recommended_classes) >= 3:
                    break

        # Summaries for the recommendations
        predefined_summaries = {
            "Equities": "Equities provide global growth opportunities by investing in companies worldwide.",
            "Fixed Income": "Fixed Income ensures capital preservation with stable returns, ideal for risk-averse investors.",
            "Real Estate": "Real Estate offers consistent income and protects against inflation through property investments.",
            "Commodities": "Commodities diversify portfolios and hedge against inflation with investments in raw materials.",
            "Alternative Investments": "Alternative Investments explore innovative growth strategies through private equity and hedge funds.",
            "Money Market": "Money Market ensures liquidity with minimal risk through Treasury bills and high-quality debt.",
            "Currencies": "Currencies diversify portfolios with exposure to global markets and hedge against forex risks.",
        }

        # Display recommendations
        st.markdown("## Recommended Asset Classes")
        for cls in recommended_classes:
            st.markdown(
                f"""
                <div style='background-color: #ADD8E6; padding: 10px; margin: 10px 0; border-radius: 5px;'>
                    <h4 style='color: #00008B;'>{cls}</h4>
                    <p style='font-size: 14px; color: black;'>{predefined_summaries[cls]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Prepare client data for the PDF
        client_data = {
            "name": client_name,
            "investment_amount": investment_amount,
            "risk_tolerance": risk_tolerance,
            "time_horizon": time_horizon,
            "objectives": objectives,
            "interests": interests,
            "rebalance_frequency": rebalance_frequency,
            "recommended_classes": recommended_classes,
            "summaries": predefined_summaries,
        }

        # Generate PDF
        pdf_data = generate_pdf_report(client_data)

        # Provide download link for the PDF
        st.download_button(
            label="Download Client Report as PDF",
            data=pdf_data,
            file_name=f"{client_name}_investment_report.pdf",
            mime="application/pdf",
        )




def display_image(image_url, caption, width=150):
    """
    Display an image in Streamlit with a circular mask applied.

    Args:
        image_url (str): URL to the image.
        caption (str): Caption for the image.
        width (int, optional): Width of the image. Defaults to 150.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Check for errors in the response
        img = Image.open(BytesIO(response.content))

        # Convert to black and white and apply circular mask
        img = img.convert("L")  # Convert to grayscale
        size = min(img.size)  # Make the image square
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)  # Create a circular mask

        img = img.crop((0, 0, size, size))  # Crop to square dimensions
        img.putalpha(mask)  # Add transparency outside the circle

        # Display the processed image in Streamlit
        st.image(img, caption=caption, width=width)
    except Exception as e:
        st.error(f"Could not load or process image: {e}")


def managers():
    st.markdown('<h2 class="section-title">Meet the Team</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 16px; line-height: 1.6; text-align: justify; color: #000;">
        We are a team of Master's students in our second year at HEC Lausanne, specializing in Asset and Risk Management. 
        Our diverse expertise and collaborative approach allow us to deliver cutting-edge insights and innovative strategies 
        tailored to the financial world.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Team members data
    team_members = [
        {
            "name": "Andre Ferreira Goncalves",
            "bio": "Andre is focused on portfolio optimization and risk analysis, bringing a data-driven approach to decision-making.",
            "image_url": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Andre%CC%81.jpeg",
        },
        {
            "name": "Matheo Good",
            "bio": "Matheo specializes in financial modeling and quantitative strategies, ensuring robust and effective solutions.",
            "image_url": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Mattheo.jpeg",
        },
        {
            "name": "Beatriz Silva Costa",
            "bio": "Beatriz excels in client relations, combining analytical skills with a client-first mindset for tailored solutions.",
            "image_url": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Bea.jpeg",
        },
        {
            "name": "Marina dos Santos de Oliveira",
            "bio": "Marina is dedicated to advanced risk modeling and asset allocation strategies to meet complex market challenges.",
            "image_url": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Marina.jpeg",
        },
        {
            "name": "Aksel Oztas",
            "bio": "Aksel focuses on global financial markets and strategic investment planning, ensuring long-term portfolio growth.",
            "image_url": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Aksel.jpeg",
        },
    ]

    # Display team members in a grid (2 per row)
    for i in range(0, len(team_members), 2):  # Loop through in steps of 2
        cols = st.columns(2)  # Create two columns per row
        for j, col in enumerate(cols):
            if i + j < len(team_members):
                member = team_members[i + j]
                with col:
                    # Display the image and bio
                    display_image(member["image_url"], caption=member["name"])
                    st.write(member["bio"])











def fees_and_minimums():

    # Display minimum investment prominently
    st.markdown(
        """
        <div style="
            background-color: #ADD8E6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            font-size: 24px;
            font-weight: bold;
            color: #00008B;
            margin-bottom: 20px;
        ">
            Minimum Investment: <span style="font-size: 36px;">CHF5,000,000</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add space between sections
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # Display investment restrictions
    st.markdown(
        """
        <div style="
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            font-size: 18px;
            color: #000;
            margin-bottom: 20px;
        ">
            <ul style="list-style-type: disc; padding-left: 20px;">
                <li>You must invest in at least <strong>3 asset classes</strong>.</li>
                <li>The minimum investment duration is <strong>1 year</strong>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add space between sections
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # Display fees as a table
    fees = {
        "Equities": 0.001,
        "Fixed Income": 0.0005,
        "Real Estate": 0.005,
        "Commodities": 0.002,
        "Alternative Investments": 0.005,
        "Money Market": 0.0002,
        "Currencies": 0.001,
    }

    # Convert fees data to DataFrame
    fees_df = pd.DataFrame(list(fees.items()), columns=["Asset Class", "Fees"])

    # Format the fees column to show percentage with two decimal places
    fees_df["Fees"] = (fees_df["Fees"] * 100).apply(lambda x: f"{x:.2f}%")

    # Set the "Asset Class" as the index
    fees_df.set_index("Asset Class", inplace=True)

    # ======= Custom CSS for Styling =======
    st.markdown(
        """
        <style>
        .dataframe tbody tr:hover {
            background-color: #d1e4f1;
        }
        .dataframe thead th {
            background-color: #00509E;
            color: white;
            font-size: 20px;
        }
        .dataframe tbody td {
            background-color: #ADD8E6;
            color: black;
            font-size: 18px;
        }
        .dataframe {
            border-radius: 8px;
            font-family: Arial, sans-serif;
            font-size: 18px;
        }
        .dataframe tbody td, .dataframe thead th {
            padding: 10px;
            text-align: center;
        }
        .section-title {
            font-size: 26px;
            font-weight: bold;
            color: #333;
        }
        /* Custom style for the table title */
        .table-title {
            font-weight: bold;
            color: light-blue;
            text-align: left;
            font-size: 22px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



    # Render the fees table in Streamlit
    st.table(fees_df)


def fetch_google_news(keyword):
    """
    Fetch 1-2 articles from Google News RSS feed based on a keyword.

    Args:
        keyword (str): Keyword to search for in Google News.

    Returns:
        list: List of news articles as dictionaries with 'title' and 'link'.
    """
    base_url = "https://news.google.com/rss/search?q="
    encoded_keyword = keyword.replace(" ", "+")  # Replace spaces with '+'
    url = f"{base_url}{encoded_keyword}"
    response = requests.get(url)

    if response.status_code != 200:
        return [f"Error fetching news for {keyword}: HTTP {response.status_code}"]

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item", limit=2)  # Limit to 1-2 articles
    articles = [{"title": item.title.text, "link": item.link.text} for item in items]

    return articles if articles else ["No news available."]

def news():


    # Display the image
    st.image(
        "https://images.pexels.com/photos/1736366/pexels-photo-1736366.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        use_container_width=True,
    )

    # Custom caption with larger font
    st.markdown(
        """
        <div style="text-align: center; font-size: 30px; margin-top: 10px;">
            Global Market Trends
        </div>
        """,
        unsafe_allow_html=True,
    )

    asset_classes = [
        "Equities",
        "Fixed Income",
        "Real Estate",
        "Commodities",
        "Alternative Investments",
        "Money Market",
        "Currencies",
    ]

    # Fetch and display 1-2 articles for each asset class
    for asset_class in asset_classes:
        st.markdown(f"### {asset_class}")
        articles = fetch_google_news(asset_class)
        for article in articles:
            if isinstance(article, dict):  # Expected article format
                st.markdown(f"- [{article['title']}]({article['link']})")
            else:  # Fallback for errors
                st.markdown(f"- {article}")


# ======== Page Selection Logic ========
if selected_page == "About Us":
    about_us()
elif selected_page == "Portfolio Characteristics":
    portfolio_characteristics()
elif selected_page == "Client Profile":
    client_profile()



elif selected_page == "Portfolio Performance":
    st.markdown("## Portfolio Performance")

    # Custom color palette for blue tones
    custom_colors = [
        "#afeeee",  # paleturquoise
        "#008080",  # teal
        "#00ffff",  # aqua
        "#5f9ea0",  # cadetblue
        "#add8e6",  # lightblue
        "#00bfff",  # deepskyblue
        "#4682b4",  # steelblue
        "#1e90ff",  # dodgerblue
        "#2f4f4f"   # darkslategray
    ]

    # Investor input variables
    st.markdown("### Investor Inputs")

    # Input for asset classes
    investor_selected_assets = st.multiselect(
        "Select Asset Classes to Include:",
        options=["Equities", "Fixed Income", "Real Estate", "Commodities", "Alternative Investments", "Currencies",
                 "Money Market"],
        default=["Equities", "Alternative Investments", "Commodities"]
    )

    # Input for rebalancing frequency
    rebalancing_freq_months = st.selectbox(
        "Select Rebalancing Frequency (Months):",
        options=[3, 6, 12],
        index=1  # Default to 6 months
    )

    # Input for investment dates
    start_date = st.date_input(
        "Select Start Date:",
        value=pd.to_datetime("2021-06-15"),
        min_value=pd.to_datetime("2017-01-01"),
        max_value=pd.to_datetime("2024-10-28")
    )
    end_date = st.date_input(
        "Select End Date:",
        value=pd.to_datetime("2024-10-28"),
        min_value=pd.to_datetime("2017-01-01"),
        max_value=pd.to_datetime("2024-10-28")
    )

    # Input for investment amount
    investment_amount = st.number_input(
        "Enter the Investment Amount (CHF):",
        min_value=5000000.0,  # Minimum investment amount is $5 million
        value=5000000.0,  # Default investment amount
        step=100000.0  # Allow custom steps of $100,000
    )

    # Validation checks
    if len(investor_selected_assets) < 3:
        st.error("You must select at least 3 asset classes to proceed.")
        st.stop()

    if pd.to_datetime(start_date) < pd.to_datetime("2017-01-01") or pd.to_datetime(end_date) > pd.to_datetime(
            "2024-10-28"):
        st.error("Date range must be between 2017-01-01 and 2024-10-28.")
        st.stop()

    # Load data with the selected rebalancing frequency
    with st.spinner("Loading data..."):
        RF, Benchs, risk_free_rate_series, market_returns, file_paths, equal_weighted_portfolios = load_all_data2(
            rebalancing_freq_months
        )
        Benchs_daily_returns = Benchs.pct_change()

    # Button to generate the portfolio
    if st.button("Generate Portfolio"):
        # Generate the ERC portfolio based on investor inputs
        with st.spinner("Generating portfolio..."):
            erc_portfolio_df = generate_erc_portfolio(
                equal_weighted_portfolios,
                start_date=start_date,
                end_date=end_date,
                rebalancing_freq_months=rebalancing_freq_months,
                investor_selected_assets=investor_selected_assets
            )

        if not erc_portfolio_df.empty:
            # Calculate fees in dollars
            erc_portfolio_df["Fees in CHF"] = erc_portfolio_df["Fees at Rebalancing"] * investment_amount

            # Display the table with performance and fees
            #st.markdown("### Portfolio Rebalancing Table")
            #st.dataframe(erc_portfolio_df[[
            #    "Rebalance Date",
            #    "Net of Fees Portfolio Return (Annualized)",
            #    "Portfolio Volatility (Annualized)",
            #    "Fees in Dollars",
            #    "Selected Classes"
            #]])

            # Portfolio Performance for a Specific Period
            st.markdown("### Portfolio Performance for a Specific Period")

            # Check if the portfolio DataFrame is not empty
            if not erc_portfolio_df.empty:
                # Compute Fees in CHF
                erc_portfolio_df["Fees in CHF"] = erc_portfolio_df["Fees at Rebalancing"] * investment_amount

                # Create the performance DataFrame for visualization
                performance_data = []
                cumulative_return = 1  # Initialize cumulative return for the overall period

                for _, row in erc_portfolio_df.iterrows():
                    rebalance_date = row["Rebalance Date"]
                    portfolio_return = row["Net of Fees Portfolio Return (Annualized)"]
                    portfolio_volatility = row["Portfolio Volatility (Annualized)"]
                    fees_chf = row["Fees in CHF"]

                    # Calculate Performance in CHF
                    performance_chf = portfolio_return * investment_amount

                    cumulative_return *= (1 + portfolio_return / 252) ** 252  # Update cumulative return

                    performance_data.append({
                        "Rebalance Date": rebalance_date,
                        "Portfolio Return (%)": portfolio_return * 100,
                        "Performance (CHF)": performance_chf,
                        "Portfolio Volatility (%)": portfolio_volatility * 100,
                        "Fees (CHF)": fees_chf,
                    })

                # Convert to DataFrame
                performance_df = pd.DataFrame(performance_data)

                # --- Interactive Plot 1: Returns ---
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Bar(
                    x=performance_df["Rebalance Date"],
                    y=performance_df["Portfolio Return (%)"],
                    name="Portfolio Return (%)",
                    marker=dict(color="blue"),
                    text=performance_df["Portfolio Return (%)"].round(2),
                    textposition="auto"
                ))
                fig_returns.update_layout(
                    title="Portfolio Returns for Each Rebalancing Period",
                    xaxis_title="Rebalance Date",
                    yaxis_title="Return (%)",
                    template="plotly_white",
                    hovermode="x unified",
                    legend_title="Legend"
                )

                # --- Interactive Plot 2: Fees ---
                fig_fees = go.Figure()
                fig_fees.add_trace(go.Bar(
                    x=performance_df["Rebalance Date"],
                    y=performance_df["Fees (CHF)"],
                    name="Fees (CHF)",
                    marker=dict(color="orange"),
                    text=performance_df["Fees (CHF)"].round(2),
                    textposition="auto"
                ))
                fig_fees.update_layout(
                    title="Fees (CHF) for Each Rebalancing Period",
                    xaxis_title="Rebalance Date",
                    yaxis_title="Fees (CHF)",
                    template="plotly_white",
                    hovermode="x unified",
                    legend_title="Legend"
                )

                # --- Interactive Plot 3: Performance in CHF ---
                fig_performance_chf = go.Figure()
                fig_performance_chf.add_trace(go.Bar(
                    x=performance_df["Rebalance Date"],
                    y=performance_df["Performance (CHF)"],
                    name="Performance (CHF)",
                    marker=dict(color="green"),
                    text=performance_df["Performance (CHF)"].round(2),
                    textposition="auto"
                ))
                fig_performance_chf.update_layout(
                    title="Portfolio Performance (CHF) for Each Rebalancing Period",
                    xaxis_title="Rebalance Date",
                    yaxis_title="Performance (CHF)",
                    template="plotly_white",
                    hovermode="x unified",
                    legend_title="Legend"
                )

                # Display plots
                st.plotly_chart(fig_returns, use_container_width=True)
                st.plotly_chart(fig_fees, use_container_width=True)
                st.plotly_chart(fig_performance_chf, use_container_width=True)

                # Compute overall metrics
                avg_annualized_return = performance_df["Portfolio Return (%)"].mean() / 100
                avg_annualized_volatility = performance_df["Portfolio Volatility (%)"].mean() / 100
                overall_cumulative_return = ((cumulative_return - 1) * 100)
                total_fees_chf = performance_df["Fees (CHF)"].sum()
                total_performance_chf = performance_df["Performance (CHF)"].sum()

                # Create annualized metrics DataFrame
                annualized_metrics_df = pd.DataFrame({
                    "Metric": ["Cumulative Return (%)", "Average Annualized Return (%)",
                               "Average Annualized Volatility (%)", "Total Fees (CHF)", "Total Performance (CHF)"],
                    "Value": [
                        f"{overall_cumulative_return:.2f}%",
                        f"{avg_annualized_return:.2%}",
                        f"{avg_annualized_volatility:.2%}",
                        f"{total_fees_chf:,.2f} CHF",
                        f"{total_performance_chf:,.2f} CHF"
                    ]
                })

                # Display annualized metrics table
                st.markdown("#### Annualized Metrics Summary")
                st.dataframe(annualized_metrics_df)

            else:
                st.warning("The portfolio data is empty. No performance to display.")

            col1, col2 = st.columns(2)

            # Column 1: Asset Allocation
            with col1:
                st.markdown("### Asset Allocation Over Time")
                weights_over_time_df = extract_weights_over_time(erc_portfolio_df)
                if not weights_over_time_df.empty:
                    # Create an interactive area chart with Plotly
                    fig_allocation = px.area(
                        weights_over_time_df,
                        x=weights_over_time_df.index,
                        y=weights_over_time_df.columns,
                        labels={"value": "Weight (%)", "variable": "Asset Classes"},
                        title="",
                    )
                    fig_allocation.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Portfolio Weight (%)",
                        legend_title="Asset Classes",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_allocation, use_container_width=True)
                else:
                    st.warning("No asset allocation data available.")

            # Column 2: Risk Contribution
            with col2:
                st.markdown("### Risk Contribution Over Time")
                risk_contributions_over_time_df = extract_risk_contributions_over_time(erc_portfolio_df)
                if not risk_contributions_over_time_df.empty:
                    # Create an interactive stacked bar chart with Plotly
                    risk_contributions_over_time_df = risk_contributions_over_time_df.reset_index()
                    fig_risk = px.bar(
                        risk_contributions_over_time_df,
                        x="index",
                        y=risk_contributions_over_time_df.columns[1:],
                        labels={"value": "Risk Contribution (%)", "index": "Date"},
                        title="",
                        barmode="stack",
                    )
                    fig_risk.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Risk Contribution (%)",
                        legend_title="Asset Classes",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                else:
                    st.warning("No risk contribution data available.")

            # Calculate and align daily returns
            with st.spinner("Aligning daily returns for visualization..."):
                portfolio_daily_returns_df = calculate_daily_returns_with_fees(
                    erc_portfolio_df=erc_portfolio_df,
                    equal_weighted_portfolios=equal_weighted_portfolios,
                    start_date=start_date,
                    end_date=end_date,
                    months=rebalancing_freq_months
                )
                aligned_daily_returns_df, aligned_market_returns = align_daily_returns(
                    portfolio_daily_returns_df,
                    Benchs_daily_returns["AQR MULTI-ASSET FUND I"]
                )

            # Interactive Cumulative Returns and Drawdowns
            st.markdown("### Cumulative Returns and Drawdowns")

            # Generate the interactive plots for cumulative returns and drawdowns
            fig_cumulative, fig_drawdowns = plot_interactive_cumulative_and_drawdowns(
                aligned_daily_returns_df=aligned_daily_returns_df,
                aligned_market_returns=aligned_market_returns
            )

            # Display the interactive plots
            st.plotly_chart(fig_cumulative, use_container_width=True)
            st.plotly_chart(fig_drawdowns, use_container_width=True)

            #--------------------------------------------------------------------



            # Portfolio vs. Benchmark Returns Interactive Plot
            st.markdown("### Our Portfolio vs. Benchmark (AQR MULTI-ASSET FUND I) Returns ")


            # Generate the comparison data for returns
            def generate_returns_data(erc_portfolios_df, benchmark_returns):
                """
                Generate data comparing portfolio returns and benchmark returns.

                Args:
                    erc_portfolios_df (pd.DataFrame): Portfolio performance metrics at rebalancing dates.
                    benchmark_returns (pd.Series): Daily returns of the benchmark.

                Returns:
                    pd.DataFrame: Comparison table of returns for each rebalancing date.
                """
                table_data = []
                for _, row in erc_portfolios_df.iterrows():
                    rebalance_date = row["Rebalance Date"]
                    portfolio_return = row["Net of Fees Portfolio Return (Annualized)"]

                    # Filter benchmark returns for the rebalancing period
                    benchmark_period = benchmark_returns.loc[
                                       rebalance_date: rebalance_date + pd.DateOffset(months=6)
                                       ].dropna()

                    if not benchmark_period.empty:
                        benchmark_return = benchmark_period.mean() * 252

                        # Append row to table
                        table_data.append({
                            "Rebalance Date": rebalance_date.date(),
                            "Portfolio Return (%)": portfolio_return * 100,
                            "Benchmark Return (%)": benchmark_return * 100,
                        })

                # Convert to DataFrame
                returns_table = pd.DataFrame(table_data)
                return returns_table


            # Generate the returns data
            returns_data = generate_returns_data(
                erc_portfolio_df, Benchs_daily_returns["AQR MULTI-ASSET FUND I"]
            )

            # Plot interactive bar chart with Plotly
            if not returns_data.empty:
                fig = go.Figure()

                # Add Portfolio Returns Bar
                fig.add_trace(go.Bar(
                    x=returns_data["Rebalance Date"],
                    y=returns_data["Portfolio Return (%)"],
                    name="Portfolio Return (%)",
                    marker_color="blue",
                    hovertemplate="Rebalance Date: %{x}<br>Portfolio Return: %{y:.2f}%<extra></extra>"
                ))

                # Add Benchmark Returns Bar
                fig.add_trace(go.Bar(
                    x=returns_data["Rebalance Date"],
                    y=returns_data["Benchmark Return (%)"],
                    name="Benchmark Return (%)",
                    marker_color="orange",
                    hovertemplate="Rebalance Date: %{x}<br>Benchmark (AQR MULTI-ASSET FUND I) Return: %{y:.2f}%<extra></extra>"
                ))

                # Update layout
                fig.update_layout(
                    title="",
                    xaxis_title="Rebalance Date",
                    yaxis_title="Return (%)",
                    barmode="group",
                    template="plotly_white",
                    hovermode="x"
                )

                # Render plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No returns data available for the selected period.")












elif selected_page == "Managers":
    managers()
elif selected_page == "Fees and Minimums":
    fees_and_minimums()
elif selected_page == "News":
    news()
elif selected_page == "Contact Us":
    st.markdown("## Contact Us")
    st.markdown(
        "We would love to hear from you. If you have any questions or inquiries, feel free to contact us at the email addresses below:")

    # Display email address
    st.markdown(
        """
        **Email Addresses:**  
        [aksel.oztas@unil.ch](mailto:aksel.oztas@unil.ch)
        [andre.ferreiragoncalves@unil.ch](mailto:andre.ferreiragoncalves@unil.ch)
        [beatriz.silvacosta@unil.ch](mailto:beatriz.silvacosta@unil.ch)
        [matheo.good@unil.ch](mailto:matheo.good@unil.ch)
        [marina.dossantosdeoliveira@unil.ch](mailto:marina.dossantosdeoliveira@unil.ch)
        """
    )

    st.markdown("Alternatively, you can fill out the form below to share your feedback or questions.")

    # Input fields for user to write feedback or inquiries
    name = st.text_input("Your Name", placeholder="Enter your name")
    email = st.text_input("Your Email", placeholder="Enter your email address")
    subject = st.text_input("Subject", placeholder="Enter the subject of your message")
    message = st.text_area("Message", placeholder="Write your message here...")

    if st.button("Submit"):
        if not name or not email or not subject or not message:
            st.error("Please fill out all fields before submitting.")
        else:
            st.success(
                "Thank you for reaching out! You can also contact us directly at [matheo.good@unil.ch](mailto:matheo.good@unil.ch).")

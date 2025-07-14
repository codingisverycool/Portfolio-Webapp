from flask import Flask, render_template, request, redirect, url_for, session, flash, get_flashed_messages 
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import squarify as sq
import io
import os
import base64
import json
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_mail import Mail, Message
import secrets
import string

app = Flask(__name__)
app.secret_key = APPKey
# Email configuration
# Email configuration (updated)
app.config['MAIL_SERVER'] = 'smtp.sendgrid.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'apikey'  # ← literally this string!
app.config['MAIL_PASSWORD'] = 'SendgridKey'
app.config['MAIL_DEFAULT_SENDER'] = 'aaryanjthaker@gmail.com'
mail = Mail(app)


# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Updated User class
class User(UserMixin):
    def __init__(self, email):
        users = load_users()
        user_data = users.get(email, {})
        self.id = email
        self.email = email
        self.verified = user_data.get('verified', False)

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# User data storage
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def generate_verification_token(length=32):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

@app.template_filter('format_number')
def format_number(value):
    if value == 'N/A':
        return value
    try:
        value = float(value)
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value:,.2f}"
    except:
        return value

@app.template_filter('format_percentage')
def format_percentage(value):
    if value == 'N/A':
        return value
    try:
        return f"{float(value)*100:.2f}%"
    except:
        return value

def get_stock_info(ticker):
    """Fetches comprehensive stock info using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Essential info
        price = stock.history(period="1d")['Close'].iloc[-1]
        currency = info.get('currency', 'N/A')
        exchange = info.get('exchange', 'N/A')
        industry = info.get('industry', 'N/A')
        sector = info.get('sector', 'N/A')
        
        # 52-week high/low
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 'N/A')
        
        return {
            'price': price,
            'currency': currency,
            'exchange': exchange,
            'industry': industry,
            'sector': sector,
            '52w_high': fifty_two_week_high,
            '52w_low': fifty_two_week_low
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = load_users()
        email = request.form['email'].strip().lower() 
        password = request.form['password']

        if email not in users:
            return render_template('login.html', error="Invalid credentials")

        user_data = users.get(email)

        if not user_data.get('verified', False):
            return render_template('login.html',
                                   error="Account not verified. Please check your email for verification link.")

        if not check_password_hash(user_data['password'], password):
            return render_template('login.html', error="Invalid credentials")

        user = User(email)
        login_user(user)
        return redirect(url_for('portfolio_tracker'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    get_flashed_messages()
    if request.method == 'POST':
        users = load_users()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validation
        if email in users:
            return render_template('register.html', error="Email already registered.")
        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match.")
        if len(password) < 8:
            return render_template('register.html', error="Password must be at least 8 characters.")

        verification_token = secrets.token_urlsafe(32)

        # Store user
        users[email] = {
            'password': generate_password_hash(password),
            'email': email,
            'verified': False,
            'verification_token': verification_token,
            'created_at': datetime.now().isoformat()
        }
        save_users(users)

        try:
            verification_link = url_for('verify_email', token=verification_token, _external=True)
            msg = Message("Verify Your Email - Portfolio Tracker", recipients=[email])
            msg.body = f"""Welcome to Portfolio Tracker!

Please verify your email:
{verification_link}

This link expires in 24 hours."""
            
            mail.send(msg)
            flash("Verification email sent! Please check your inbox.", "success")
            return render_template('register.html', 
                success="Registration successful! Please check your email to verify your account.")
        
        except Exception as e:
            # Full traceback + log
            import traceback
            traceback.print_exc()
            print(f"Failed to send verification email: {e}")

            # Remove user from file
            users.pop(email, None)
            save_users(users)

            return render_template('register.html', 
                error="Failed to send verification email. Make sure you're using a working Gmail App Password.")

    return render_template('register.html')

@app.route('/verify/<token>')
def verify_email(token):
    users = load_users()
    for username, user_data in users.items():
        if user_data.get('verification_token') == token:
            # Check if the token is expired (24 hours)
            created_at = datetime.fromisoformat(user_data.get('created_at', ''))
            if (datetime.now() - created_at).days > 1:
                return render_template('verify.html',
                                       error="Verification link has expired. Please register again.")

            # Mark the user as verified
            user_data['verified'] = True
            user_data.pop('verification_token', None)
            save_users(users)
            return render_template('verify.html',
                                   success="Email verified successfully! You can now login.")
    
    return render_template('verify.html',
                           error="Invalid verification link.")

def get_user_data_file():
    return f'user_data/{current_user.id}/portfolio.json'

def load_user_data():
    try:
        with open(get_user_data_file()) as f:
            data = json.load(f)
            # Convert back to Flask's MultiDict-like structure
            return {
                'stock[]': data.get('stock[]', []),
                'quantity[]': data.get('quantity[]', []),
                'investment_date[]': data.get('investment_date[]', []),
                'avg_cost[]': data.get('avg_cost[]', [])
            }
    except:
        return None

def save_user_data(data):
    # Ensure directory exists
    os.makedirs(os.path.dirname(get_user_data_file()), exist_ok=True)
    # Save as plain dictionary
    with open(get_user_data_file(), 'w') as f:
        json.dump({
            'stock[]': data.getlist('stock[]') if hasattr(data, 'getlist') else data.get('stock[]', []),
            'quantity[]': data.getlist('quantity[]') if hasattr(data, 'getlist') else data.get('quantity[]', []),
            'investment_date[]': data.getlist('investment_date[]') if hasattr(data, 'getlist') else data.get('investment_date[]', []),
            'avg_cost[]': data.getlist('avg_cost[]') if hasattr(data, 'getlist') else data.get('avg_cost[]', [])
        }, f)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def portfolio_tracker():
    if request.method == 'POST':
        try:
            # Get all form data
            stocks = request.form.getlist('stock[]')
            quantities = request.form.getlist('quantity[]')
            investment_dates = request.form.getlist('investment_date[]')
            avg_costs = request.form.getlist('avg_cost[]')

            structured_form_data = {
                'stock[]': stocks,
                'quantity[]': quantities,
                'investment_date[]': investment_dates,
                'avg_cost[]': avg_costs
            }
            save_user_data(structured_form_data)
            
            # Validate inputs
            if not stocks or not quantities or not investment_dates or not avg_costs:
                raise ValueError("Please fill in all fields")
                
            # Process portfolio data - combine duplicates
            portfolio_dict = {}
            for stock, quantity, date, avg_cost in zip(stocks, quantities, investment_dates, avg_costs):
                if not stock or not quantity or not date or not avg_cost:
                    continue
                    
                try:
                    stock = stock.upper()
                    quantity = int(quantity)
                    if quantity <= 0:
                        raise ValueError("Quantity must be positive")
                        
                    avg_cost = float(avg_cost)
                    if avg_cost <= 0:
                        raise ValueError("Average cost must be positive")
                        
                    # Validate date format
                    try:
                        datetime.strptime(date, '%Y-%m-%d')
                    except ValueError:
                        raise ValueError("Invalid date format. Use YYYY-MM-DD")
                        
                    # Get stock info
                    stock_info = get_stock_info(stock)
                    if stock_info is None:
                        raise ValueError(f"Could not fetch data for {stock}")
                        
                    # Combine duplicate stocks
                    if stock in portfolio_dict:
                        existing = portfolio_dict[stock]
                        total_qty = existing['Quantity'] + quantity
                        # Calculate weighted average cost
                        new_avg_cost = ((existing['Quantity'] * existing['Avg Cost/Share']) + 
                                       (quantity * avg_cost)) / total_qty
                        # Keep earliest investment date
                        new_date = min(existing['Investment Date'], date)
                        
                        portfolio_dict[stock] = {
                            **existing,
                            'Quantity': total_qty,
                            'Avg Cost/Share': new_avg_cost,
                            'Total Cost': total_qty * new_avg_cost,
                            'Investment Date': new_date
                        }
                    else:
                        portfolio_dict[stock] = {
                            'Stock': stock,
                            'Quantity': quantity,
                            'Current Price': stock_info['price'],
                            'Currency': stock_info['currency'],
                            'Exchange': stock_info['exchange'],
                            'Industry': stock_info['industry'],
                            'Sector': stock_info['sector'],
                            '52w High': stock_info['52w_high'],
                            '52w Low': stock_info['52w_low'],
                            'Investment Date': date,
                            'Avg Cost/Share': avg_cost,
                            'Total Cost': quantity * avg_cost
                        }
                except Exception as e:
                    raise ValueError(f"Error processing {stock}: {str(e)}")
            
            if not portfolio_dict:
                raise ValueError("No valid stocks entered")
            
            # Store tickers in session for valuation dashboard
            session['portfolio_tickers'] = list(portfolio_dict.keys())
            session['portfolio_data'] = portfolio_dict  # Save the raw data too

            # Create DataFrame and calculate metrics
            portfolio_df = pd.DataFrame(portfolio_dict.values())
            current_date = datetime.now().strftime('%Y-%m-%d')
            portfolio_df['Current Date'] = current_date
            
            # Calculate performance metrics
            portfolio_df['Current Value'] = portfolio_df['Quantity'] * portfolio_df['Current Price']
            portfolio_df['Gain/Loss'] = portfolio_df['Current Value'] - portfolio_df['Total Cost']
            portfolio_df['Gain/Loss %'] = (portfolio_df['Gain/Loss'] / portfolio_df['Total Cost']) * 100
            
            # Calculate distance from 52-week high/low
            portfolio_df['% From 52w High'] = ((portfolio_df['Current Price'] - portfolio_df['52w High']) / 
                                             portfolio_df['52w High']) * 100
            portfolio_df['% From 52w Low'] = ((portfolio_df['Current Price'] - portfolio_df['52w Low']) / 
                                            portfolio_df['52w Low']) * 100
            
            total_value = portfolio_df['Current Value'].sum()
            total_cost = portfolio_df['Total Cost'].sum()
            total_gain = total_value - total_cost
            total_gain_pct = (total_gain / total_cost) * 100
            
            portfolio_df['Allocation (%)'] = (portfolio_df['Current Value'] / total_value) * 100
            
            # Reorder columns for better display
            columns_order = [
                'Stock', 'Currency', 'Exchange', 'Industry', 'Sector',
                'Quantity', 'Investment Date', 'Current Date', 
                'Avg Cost/Share', 'Current Price', '52w High', '52w Low',
                '% From 52w High', '% From 52w Low', 'Total Cost', 
                'Current Value', 'Gain/Loss', 'Gain/Loss %', 'Allocation (%)'
            ]
            portfolio_df = portfolio_df[columns_order]
            
            # Generate all visualizations
            img_data = {}
            
            # 1. Stock Allocation Pie Chart
            plt.figure(figsize=(8, 6))
            plt.pie(portfolio_df['Allocation (%)'], 
                   labels=portfolio_df['Stock'], 
                   autopct='%1.1f%%', 
                   startangle=140)
            plt.title("Stock Allocation")
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            img_data['stock_pie'] = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()
            
            # 2. Sector Allocation Pie Chart
            sector_values = portfolio_df.groupby('Sector')['Current Value'].sum()
            plt.figure(figsize=(8, 6))
            plt.pie(sector_values,
                   labels=sector_values.index,
                   autopct='%1.1f%%',
                   startangle=140)
            plt.title("Sector Allocation")
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            img_data['sector_pie'] = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()
            
            # 3. Treemap of Current Value (color by gain/loss)
            plt.figure(figsize=(12, 8))
            
            # Generate colors based on gain/loss percentage
            colors = []
            max_gain = max(portfolio_df['Gain/Loss %'].max(), 0)  # Ensure non-negative
            max_loss = min(portfolio_df['Gain/Loss %'].min(), 0)  # Ensure non-positive

            for _, row in portfolio_df.iterrows():
                if row['Gain/Loss %'] > 0:
                    # Darker green for higher gains
                    intensity = 1.0 - 0.7 * (row['Gain/Loss %'] / max_gain if max_gain > 0 else 0)
                    colors.append((0, 0.3 + intensity*0.7, 0, 0.7))
                else:
                    # Darker red for higher losses
                    intensity = 0.3 + 0.7 * (abs(row['Gain/Loss %']) / abs(max_loss) if max_loss < 0 else 0)
                    colors.append((intensity, 0, 0, 0.7))
            
            sq.plot(
                sizes=portfolio_df['Current Value'],
                label=[f"{row['Stock']}\n${row['Current Value']/1000:.1f}k" for _, row in portfolio_df.iterrows()],
                color=colors,
                alpha=0.7,
                text_kwargs={'fontsize':8}
            )
            plt.title("Portfolio Treemap (Size = Current Value, Color = Gain/Loss)")
            plt.axis('off')
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            img_data['treemap'] = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()

            session['portfolio_tickers'] = list(portfolio_dict.keys())
            session['portfolio_data'] = portfolio_df.set_index('Stock').to_dict('index')
            
            return render_template('index.html', 
                                table=portfolio_df.to_html(classes='data', index=False, float_format='%.2f'),
                                total_value=f"${total_value:,.2f}",
                                total_cost=f"${total_cost:,.2f}",
                                total_gain=f"${total_gain:,.2f}",
                                total_gain_pct=f"{total_gain_pct:.2f}%",
                                img_data=img_data,
                                show_results=True,
                                form_data=structured_form_data)
            
        except Exception as e:
            return render_template('index.html', 
                                 error_message=str(e),
                                 show_results=False,
                                 form_data=structured_form_data)
    
    # GET request - load saved data
    saved_data = load_user_data()
    if saved_data:
        return render_template('index.html', 
                            form_data=saved_data,
                            show_results=False)
    return render_template('index.html', show_results=False)
    

@app.route('/valuation')
@login_required
def valuation_dashboard():
    # Clear any existing portfolio data if it belongs to a different user
    if 'portfolio_user' not in session or session['portfolio_user'] != current_user.id:
        session.pop('portfolio_tickers', None)
        session.pop('portfolio_data', None)
        session['portfolio_user'] = current_user.id
        
    
    if 'portfolio_tickers' not in session:
        return redirect(url_for('portfolio_tracker'))
    
    portfolio_tickers = session['portfolio_tickers']
    portfolio_data = session.get('portfolio_data', {})
    
    # Calculate total portfolio value correctly
    total_value = sum(stock['Current Value'] for stock in portfolio_data.values())
    
    valuation_data = {}
    for ticker in portfolio_tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price from portfolio data to ensure consistency
        current_price = portfolio_data[ticker]['Current Price']
        current_value = portfolio_data[ticker]['Current Value']
        
        # Calculate correct allocation percentage
        allocation_pct = (current_value / total_value * 100) if total_value > 0 else 0
        
        valuation_data[ticker] = {
            'Currency': portfolio_data[ticker]['Currency'],
            'Current Price': current_price,
            'Current Value': current_value,
            'Allocation %': allocation_pct,
            '52w High': info.get('fiftyTwoWeekHigh', current_price),
            '52w Low': info.get('fiftyTwoWeekLow', current_price),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Trailing P/E': info.get('trailingPE', 'N/A'),
            'Forward P/E': info.get('forwardPE', 'N/A'),
            'Price/Sales': info.get('priceToSalesTrailing12Months', 'N/A'),
            'Price/Book': info.get('priceToBook', 'N/A'),
            'Debt/Equity': info.get('debtToEquity', 'N/A')/100,
            'Free Cash Flow': info.get('freeCashflow', 'N/A'),
            'PEG Ratio': info.get('pegRatio', 'N/A'),
            'Enterprise Value/Revenue': info.get('enterpriseToRevenue', 'N/A'),
            'Enterprise Value/EBITDA': info.get('enterpriseToEbitda', 'N/A'),
            
            # Financial highlights
            'Profit Margin': info.get('profitMargins', 'N/A')*100,
            'ROA': info.get('returnOnAssets', 'N/A')*100,
            'ROE': info.get('returnOnEquity', 'N/A')*100,
            'Revenue': info.get('totalRevenue', 'N/A'),
            'Net Income': info.get('netIncomeToCommon', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Total Cash': info.get('totalCash', 'N/A'),
        }
    
    return render_template('valuation.html', 
                         valuation_data=valuation_data,
                         tickers=session['portfolio_tickers'])

@app.template_filter('format_currency')
def format_currency(value, currency='USD', decimals=2):
    if value == 'N/A':
        return value
    try:
        value = float(value)
        
        # Get currency symbol
        if currency == 'USD':
            symbol = '$'
        elif currency == 'EUR':
            symbol = '€'
        elif currency == 'GBP':
            symbol = '£'
        elif currency == 'JPY':
            symbol = '¥'
        else:
            symbol = currency + ' '
        
        # Format with appropriate scale
        if abs(value) >= 1e12:
            return f"{symbol}{value/1e12:,.{decimals}f}T"
        elif abs(value) >= 1e9:
            return f"{symbol}{value/1e9:,.{decimals}f}B"
        elif abs(value) >= 1e6:
            return f"{symbol}{value/1e6:,.{decimals}f}M"
        else:
            return f"{symbol}{value:,.{decimals}f}"
    except:
        return value

@app.template_filter('format_percentage')
def format_percentage(value, decimals=2):
    if value == 'N/A':
        return value
    try:
        # Fix allocation percentage if needed
        value = float(value)
        if 'Allocation' in request.path:
            value = min(value, 100)  # Cap at 100%
        elif 'ROE' in request.path:
            return f"{value:,.1f}%"
        return f"{value:,.{decimals}f}%"
    except:
        return value

@app.template_filter('format_ratio')
def format_ratio(value):
    if value == 'N/A':
        return value
    try:
        value = float(value)
        return f"{value:,.2f}x"
    except:
        return value


if __name__ == '__main__':
    app.run(debug=True, port=8000)

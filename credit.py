import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------
# Step 1: Load JSON and normalize data
# ------------------------------------------
def load_data(json_file):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)

    df = pd.json_normalize(raw_data)
    return df

# ------------------------------------------
# Step 2: Feature Engineering
# ------------------------------------------
def feature_engineering(df):
    print("🧾 Columns in loaded DataFrame:")
    print(df.columns.tolist())

    print("📌 First row sample:")
    print(df.head(1).to_string())

    # Auto-detect wallet column
    possible_wallet_keys = [
        'user', 'wallet', 'user_id', 'address', 'user_address',
        'walletAddress', 'userwallet', 'userWallet'
    ]
    wallet_column = None
    for col in df.columns:
        if col.lower() in [k.lower() for k in possible_wallet_keys]:
            wallet_column = col
            break

    if not wallet_column:
        print("⚠️ Could not auto-detect wallet column. Setting manually.")
        wallet_column = 'userWallet'  # Set manually

    print(f"🧠 Using wallet column: {wallet_column}")

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    grouped = df.groupby(wallet_column)

    feature_data = []

    for wallet, group in grouped:
        feature = {}
        feature['user'] = wallet
        feature['total_txn'] = len(group)
        feature['unique_days_active'] = group['timestamp'].dt.date.nunique()
        feature['avg_txn_delay'] = group['timestamp'].diff().dt.total_seconds().mean()

        for action in ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']:
            sub = group[group['action'].str.lower() == action]
            amt_col = 'actionData.amount'
            if amt_col in sub.columns:
                amt_sum = pd.to_numeric(sub[amt_col], errors='coerce').fillna(0).sum()
            else:
                amt_sum = 0

            feature[f'{action}_count'] = len(sub)
            feature[f'total_{action}_amt'] = amt_sum

        # Ratios
        feature['repay_ratio'] = (
            feature['total_repay_amt'] / feature['total_borrow_amt']
            if feature['total_borrow_amt'] > 0 else 0
        )
        feature['redeem_ratio'] = (
            feature['total_redeemunderlying_amt'] / feature['total_deposit_amt']
            if feature['total_deposit_amt'] > 0 else 0
        )
        feature['borrow_to_deposit_ratio'] = (
            feature['total_borrow_amt'] / feature['total_deposit_amt']
            if feature['total_deposit_amt'] > 0 else 0
        )
        feature['liquidation_flag'] = 1 if feature['liquidationcall_count'] > 0 else 0

        feature_data.append(feature)

    return pd.DataFrame(feature_data)

# ------------------------------------------
# Step 3: Scoring Logic Based on Behavior
# ------------------------------------------
def score_wallets(features_df):
    features_df.fillna(0, inplace=True)

    df = features_df.copy()

    # 🔹 Behavior 1: Deposits + Repayments
    max_good_amt = df[['total_deposit_amt', 'total_repay_amt']].max().max()
    good_behavior_score = (
        (df['total_deposit_amt'] + df['total_repay_amt']) / (max_good_amt + 1)
    ) * 400  # Max 400 points

    # 🔹 Behavior 2: Active Days
    max_active_days = df['unique_days_active'].max()
    activity_score = (df['unique_days_active'] / (max_active_days + 1)) * 100  # Max 100 points

    # 🔸 Penalty 1: Borrow-to-Deposit Ratio
    borrow_ratio_penalty = df['borrow_to_deposit_ratio'].clip(0, 1) * 200  # Max -200 points

    # 🔸 Penalty 2: Liquidation
    liquidation_penalty = df['liquidation_flag'] * 300  # -300 points if liquidated

    # 🧠 Final Score
    score = good_behavior_score + activity_score - borrow_ratio_penalty - liquidation_penalty
    score = score.clip(0, 1000)

    df['score'] = score.astype(int)
    return df[['user', 'score']].sort_values(by='score', ascending=False)

# ------------------------------------------
# Step 4: Entry Point
# ------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to JSON file')
    parser.add_argument('--output', type=str, default='wallet_scores.csv', help='Output CSV file name')
    args = parser.parse_args()

    print("📥 Loading data...")
    raw_df = load_data(args.input)

    print("⚙️ Extracting features...")
    feature_df = feature_engineering(raw_df)

    print("📊 Scoring wallets...")
    scored_df = score_wallets(feature_df)

    print(f"💾 Saving to {args.output}")
    scored_df.to_csv(args.output, index=False)

    print("✅ Done! Output saved to:", args.output)



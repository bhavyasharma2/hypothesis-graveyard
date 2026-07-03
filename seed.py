import os
import sys
from dotenv import load_dotenv

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import graveyard

load_dotenv()

SEED_HYPOTHESES = [
    {
        "text": "Using a multi-task Transformer model predicting 10-second price trends in NASDAQ liquid stocks using Level 3 order book tick data.",
        "domain": "Quant Finance",
        "outcome": "Failed",
        "conviction_score": 35.0,
        "notes": (
            "Severe overfitting and feature drift. The model achieved 58% out-of-sample accuracy in backtests, "
            "but in live trading, the GPU inference latency of 45ms caused us to miss the execution window. "
            "When accounting for a 12ms network latency, the realized Sharpe was -1.2, and we incurred "
            "$140,000 in slippage before halting the strategy."
        ),
        "contributor": "Priya Sharma",
        "date": "2025-01-15"
    },
    {
        "text": "Using XGBoost on alternative data (satellite imagery of retail parking lots combined with credit card transaction feeds) to predict quarterly revenue surprises for major US retail tickers.",
        "domain": "Quant Finance",
        "outcome": "Success",
        "conviction_score": 80.0,
        "notes": (
            "Highly profitable. Achieved an information ratio of 1.38 and a Sharpe ratio of 1.55 over 2.5 years. "
            "The model successfully predicted 14 out of 16 retail revenue beats/misses by more than 2.5%, "
            "allowing us to build profitable long/short positions 5 days prior to earnings announcements. "
            "Net profit generated was $2.4M with a maximum drawdown of 6.2%."
        ),
        "contributor": "Alex Chen",
        "date": "2025-02-28"
    },
    {
        "text": "Dynamic delta hedging of weekly SPX options using a GARCH(1,1) volatility forecast to capture intra-day volatility smiles.",
        "domain": "Quant Finance",
        "outcome": "Failed",
        "conviction_score": 40.0,
        "notes": (
            "High transaction costs and tail risk. The bid-ask spread of 0.15% on weekly options and overnight "
            "gap risk (market opened down 2.1% on a key CPI day) wiped out the premium capture. The strategy "
            "realized a Sharpe ratio of -0.42 and a maximum drawdown of 18.5% over a 6-month trial, failing to "
            "outperform a simple buy-and-hold strategy."
        ),
        "contributor": "Marcus Webb",
        "date": "2025-03-12"
    },
    {
        "text": "Dispersion trading on the Dow Jones Industrial Average (DJIA) index options vs. constituent options during earnings season.",
        "domain": "Quant Finance",
        "outcome": "Success",
        "conviction_score": 85.0,
        "notes": (
            "Statistically robust. Successfully captured the systematic overpricing of index implied volatility "
            "compared to realized individual stock correlation. Achieved a Sharpe ratio of 1.65 over 3 years, "
            "with a maximum drawdown of 8.4% during the Q3 earnings surprise. Total return was 22.4% annualized "
            "after accounting for transaction costs."
        ),
        "contributor": "Sara Iyer",
        "date": "2025-04-19"
    },
    {
        "text": "Queue-position-based execution on CME Eurodollar futures using order book imbalance (OBI) and order flow toxicity (VPIN).",
        "domain": "Quant Finance",
        "outcome": "Failed",
        "conviction_score": 20.0,
        "notes": (
            "Suffered from severe adverse selection and latency arbitrage by HFT firms. Our execution loop "
            "had a round-trip latency of 120 microseconds, but HFT competitors reacted in under 5 microseconds. "
            "Our fill rate on favorable moves was only 12%, while getting filled 92% of the time when the price "
            "moved against us. Realized a Sharpe ratio of -1.8."
        ),
        "contributor": "Priya Sharma",
        "date": "2025-05-05"
    },
    {
        "text": "Cross-venue liquidity provisioning in crypto perpetual swaps (Binance vs. dYdX) using a stochastic control framework (Avellaneda-Stoikov).",
        "domain": "Quant Finance",
        "outcome": "Proposed",
        "conviction_score": 70.0,
        "notes": (
            "Proposing to place limit orders adjusted for inventory risk. Backtests indicate we can capture "
            "a net spread of 1.2 bps per trade while keeping inventory variance below 0.05 BTC^2. Expected "
            "Sharpe ratio is 2.1 with a maximum expected drawdown of 12% during peak volatility events. "
            "Requires co-location and <5ms API round-trip times."
        ),
        "contributor": "Alex Chen",
        "date": "2025-06-22"
    },
    {
        "text": "Statistical arbitrage of ETF-constituent pairs (e.g., XLF vs. top 5 bank stocks) using an Ornstein-Uhlenbeck mean-reversion model.",
        "domain": "Quant Finance",
        "outcome": "Failed",
        "conviction_score": 30.0,
        "notes": (
            "Cointegration relationship broke down catastrophically during the March 2023 regional banking crisis. "
            "The historical relationship diverged by over 6 standard deviations. Lack of hard stop-losses led to "
            "a loss of $4.2M, representing a 28% drawdown on the fund's capital. Sharpe ratio plummeted from "
            "1.4 to -2.1 in a single week."
        ),
        "contributor": "Marcus Webb",
        "date": "2025-08-14"
    },
    {
        "text": "Statistical arbitrage of US PJM and ERCOT power markets using neural network-based cointegration residuals.",
        "domain": "Quant Finance",
        "outcome": "Proposed",
        "conviction_score": 65.0,
        "notes": (
            "Proposing to trade virtual transactions in the day-ahead vs. real-time markets using hourly "
            "temperature forecasts. Initial simulations yield a target Sharpe of 2.4 with a maximum expected "
            "drawdown of 14% during peak summer loads. Key risk is extreme weather anomalies causing price "
            "spikes up to $5,000/MWh."
        ),
        "contributor": "Sara Iyer",
        "date": "2025-09-30"
    },
    {
        "text": "Migrating our high-throughput user activity log from Cassandra to MongoDB to support complex ad-hoc aggregation queries.",
        "domain": "Software Engineering",
        "outcome": "Failed",
        "conviction_score": 25.0,
        "notes": (
            "MongoDB's document-level locking and replica set oplog replication could not sustain our write "
            "throughput of 85,000 writes/sec. Write latency spiked from 4ms to 320ms, and CPU utilization hit "
            "100% on the primary node, leading to a 45-minute partial outage. We had to roll back to Cassandra, "
            "costing $45,000 in wasted infra."
        ),
        "contributor": "Priya Sharma",
        "date": "2025-11-08"
    },
    {
        "text": "Migrating our primary transactional ledger from a sharded MySQL setup to a multi-region Google Cloud Spanner database to achieve global ACID transactions.",
        "domain": "Software Engineering",
        "outcome": "Success",
        "conviction_score": 90.0,
        "notes": (
            "Successfully eliminated write-skew anomalies and reduced database maintenance overhead by 90%. "
            "Global write latency stabilized at 14ms (down from 45ms cross-region MySQL replication), and we "
            "achieved 99.999% uptime during Black Friday, handling a peak of 12,000 transactions per second "
            "without a single consistency failure."
        ),
        "contributor": "Alex Chen",
        "date": "2025-12-25"
    },
    {
        "text": "Migrating our monolithic API gateway to a decentralized service mesh (Istio on Kubernetes) to improve microservice communication and security.",
        "domain": "Software Engineering",
        "outcome": "Failed",
        "conviction_score": 30.0,
        "notes": (
            "Huge latency penalty and resource overhead. The sidecar proxy (Envoy) added 8.2ms of overhead per "
            "hop, which compounded to over 45ms for complex nested service calls. The Kubernetes cluster "
            "memory footprint increased by 35%, costing an extra $18,000/month with zero measurable improvement "
            "in developer velocity or security posture."
        ),
        "contributor": "Marcus Webb",
        "date": "2026-01-18"
    },
    {
        "text": "Migrating our legacy Ruby on Rails monolithic background worker pool to a Go-based distributed worker pool using temporal.io.",
        "domain": "Software Engineering",
        "outcome": "Success",
        "conviction_score": 85.0,
        "notes": (
            "Reduced background job execution latency by 82% (from 1.2s to 210ms) and slashed server hosting "
            "costs by 65% ($42,000 saved monthly). It also resolved long-standing race conditions in our "
            "order-fulfillment state machine, reducing order processing errors from 0.4% to less than 0.001%."
        ),
        "contributor": "Sara Iyer",
        "date": "2026-02-14"
    },
    {
        "text": "Implementing a two-tier caching strategy (in-memory L1 cache using Go-Cache, and a shared L2 cache using Redis) for our product catalog service.",
        "domain": "Software Engineering",
        "outcome": "Success",
        "conviction_score": 88.0,
        "notes": (
            "Cache hit rate increased from 72% to 98.4%. Database read traffic dropped by 85%, and the 99th "
            "percentile response time for catalog queries improved from 120ms to 8ms under a simulated load "
            "of 50,000 concurrent users. This reduced our database cluster size from 12 nodes to 3 nodes."
        ),
        "contributor": "Priya Sharma",
        "date": "2026-03-29"
    },
    {
        "text": "Replacing Redis with DragonflyDB to handle high-concurrency caching of session tokens.",
        "domain": "Software Engineering",
        "outcome": "Proposed",
        "conviction_score": 75.0,
        "notes": (
            "Proposing to leverage Dragonfly's multi-threaded architecture. Anticipating a 3x throughput "
            "increase (up to 1M ops/sec) and a 40% reduction in memory footprint due to its more efficient "
            "hash table implementation. This is expected to reduce our caching infrastructure costs by $8,500 "
            "per month."
        ),
        "contributor": "Alex Chen",
        "date": "2026-04-10"
    },
    {
        "text": "Moving our video transcoding pipeline from AWS EC2 spot instances to serverless AWS Lambda to scale infinitely and reduce idle server costs.",
        "domain": "Software Engineering",
        "outcome": "Failed",
        "conviction_score": 15.0,
        "notes": (
            "Extremely high cost and execution timeouts. Long-running videos (over 15 minutes) hit the "
            "15-minute Lambda timeout limit, requiring complex state saving and resume logic. The serverless "
            "bill for the first month was $38,500, compared to $12,400 when using auto-scaled EC2 spot "
            "instances, a 210% cost increase."
        ),
        "contributor": "Marcus Webb",
        "date": "2026-05-24"
    },
    {
        "text": "Using transformer models on tabular financial data to outperform gradient boosting.",
        "domain": "ML / Data Science",
        "outcome": "Failed",
        "conviction_score": 25.0,
        "notes": (
            "Severe overfitting on small datasets. The transformer's lack of inductive bias made it "
            "highly prone to memorizing noise in small financial tables. Standard gradient boosting "
            "models (XGBoost/LightGBM) consistently outperformed it in out-of-sample testing."
        ),
        "contributor": "Priya Sharma",
        "date": "2026-06-01"
    },
    {
        "text": "Replacing manual feature engineering with raw LLM embeddings for fraud detection.",
        "domain": "ML / Data Science",
        "outcome": "Failed",
        "conviction_score": 30.0,
        "notes": (
            "Distribution shift in production destroyed performance. The embeddings generated by the LLM "
            "captured static semantic relationships but failed to adapt to dynamic, evolving fraud patterns "
            "in real-time transactions, leading to a massive spike in false positives."
        ),
        "contributor": "Alex Chen",
        "date": "2026-06-10"
    },
    {
        "text": "Using GANs for data augmentation to handle class imbalance in credit default prediction.",
        "domain": "ML / Data Science",
        "outcome": "Failed",
        "conviction_score": 20.0,
        "notes": (
            "Mode collapse produced unrealistic synthetic samples. The generator failed to capture the "
            "full diversity of the minority class, generating highly repetitive and artificial profiles "
            "that misled downstream classifiers and degraded overall model sensitivity."
        ),
        "contributor": "Marcus Webb",
        "date": "2026-06-18"
    },
    {
        "text": "AutoML for hyperparameter optimization on time series forecasting.",
        "domain": "ML / Data Science",
        "outcome": "Success",
        "conviction_score": 78.0,
        "notes": (
            "15% improvement over manual tuning with minimal effort. The automated pipeline "
            "efficiently explored the search space for neural forecasting and ARIMA configurations, "
            "drastically reducing training search times and improving mean absolute scaled error (MASE)."
        ),
        "contributor": "Sara Iyer",
        "date": "2026-06-25"
    },
    {
        "text": "Transfer learning from ImageNet pretrained CNNs for satellite imagery land classification.",
        "domain": "ML / Data Science",
        "outcome": "Success",
        "conviction_score": 82.0,
        "notes": (
            "Significantly outperformed training from scratch. Despite the domain gap between natural "
            "images and top-down satellite views, the pretrained feature extractors generalized remarkably "
            "well, converging much faster and requiring 60% fewer labeled training examples."
        ),
        "contributor": "Priya Sharma",
        "date": "2026-07-01"
    }
]

def seed_database():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("ERROR: GEMINI_API_KEY is not set or is still the placeholder.")
        print("Please set your GEMINI_API_KEY in the .env file before running seed.py.")
        sys.exit(1)
        
    print("Seeding ChromaDB with historical hypotheses...")
    for hyp in SEED_HYPOTHESES:
        try:
            doc_id = graveyard.store_hypothesis(
                text=hyp["text"],
                domain=hyp["domain"],
                outcome=hyp["outcome"],
                conviction_score=hyp["conviction_score"],
                notes=hyp["notes"],
                contributor=hyp.get("contributor", "Anonymous"),
                date=hyp.get("date")
            )
            print(f"Stored: {hyp['text'][:40]}... ID: {doc_id}")
        except Exception as e:
            print(f"Error storing '{hyp['text'][:40]}...': {e}")
            
    print("\nSeeding complete! Local vector store is ready.")

if __name__ == "__main__":
    seed_database()

"""
SecConfig.py / configuration: contains the configuration details of various
different types of classes and objects used in the construction of the RAG
"""
import torch
from dataclasses import dataclass
from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Disable memory-efficient SDP and Flash SDP
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

@dataclass
class SecPlotConfig:
    """
    Data class for SEC plot configuration.
    """
    df_x: str
    df_val: str
    df_metric: str
    df_filing: str
    df_x_label: str
    filing_type: str
    stock_ticker: str

    df_cols: list[str]
    fin_metrics: list[str]


@dataclass
class SecDbConfig:
    """
    Data class for SEC database configuration.
    """
    end_date: str
    start_date: str
    secd_email: str
    filing_type: str
    secd_company: str
    stock_ticker: str
    embedding_model: str
    model_chunk_size: int
    secd_base_data_dir: str
    model_chunk_overlap: int
    embeddings_cache_dir: str
    force_recalculate_index: bool


@dataclass
class SecCliConfig:
    """
    Data class for SEC CLI configuration.
    """
    model_name: str
    tokenizer: AutoTokenizer
    bnb_config: BitsAndBytesConfig
    model: AutoModelForCausalLM
    pipe: pipeline
    hf_pipe: HuggingFacePipeline



# Default SEC plot configuration
DEFAULT_SEC_PLOT_CONFIG = SecPlotConfig(
    df_x="fy",
    df_val="val",
    df_metric="fact",
    df_filing="form",
    df_x_label="Financial Year (FY)",
    filing_type="10-K",
    stock_ticker="AAPL",
    df_cols=["end", "fy", "fp", "filed", "val"],
    fin_metrics=[
        "Cash",
        "Assets",
        "NetIncomeLoss",
        "SalesRevenueNet",
        "StockholdersEquity",
        "InventoryNet",
        "OperatingIncomeLoss",
        "LongTermDebt",
        "Liabilities",
        "EarningsPerShareBasic",
    ],
)

# Default SEC database configuration
DEFAULT_SEC_DB_CONFIG = SecDbConfig(
    end_date="2023-12-31",
    start_date="1995-01-01",
    secd_email="anshsarkar@gmail.com",
    filing_type="10-K",
    secd_company="NA",
    stock_ticker="AAPL",
    embedding_model="sentence-transformers/all-MiniLM-l6-v2",
    model_chunk_size=800,
    secd_base_data_dir="data/",
    model_chunk_overlap=150,
    embeddings_cache_dir="embeddings/",
    force_recalculate_index=False,
)

# Default model name
DEFAULT_MODEL_NAME = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"

# Default BitsAndBytesConfig for Model Quantization
DEFAULT_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Default tokenizer
DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

# Default loaded model
DEFAULT_LOADED_MODEL = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL_NAME,
    quantization_config=DEFAULT_BNB_CONFIG,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Default pipeline
DEFAULT_PIPE = pipeline(
    "text-generation",
    model=DEFAULT_LOADED_MODEL,
    tokenizer=DEFAULT_TOKENIZER,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens=1024,
)

# Default SEC CLI configuration
DEFAULT_SEC_CLI_CONFIG = SecCliConfig(
    model_name=DEFAULT_MODEL_NAME,
    tokenizer=DEFAULT_TOKENIZER,
    bnb_config=DEFAULT_BNB_CONFIG,
    model=DEFAULT_LOADED_MODEL,
    pipe=DEFAULT_PIPE,
    hf_pipe=HuggingFacePipeline(pipeline=DEFAULT_PIPE),
)

# Builtin prompts dictionary
builtin_prompts = {
    "insights": "Please provide a detailed financial analysis of a company based on its recent 10-K filings."
    + "Your analysis should cover the following financial aspects:"
    + "1. Revenue Trends: Describe the company's revenue trends over the past few "
    + "years and discuss any significant changes or patterns."
    + "2. Profitability Analysis: Evaluate the company's profitability by analyzing metrics "
    + "such as gross profit margin, operating profit margin, and net profit margin. Discuss "
    + "any factors influencing profitability."
    + "3. Cash Flow Analysis: Assess the company's cash flow from operating activities, investing "
    + "activities, and financing activities. Comment on the company's ability to generate cash "
    + "and manage its cash flow effectively."
    + "4. Debt Levels and Obligations: Review the company's debt levels, including long-term debt, current "
    + "debt, and debt repayment schedules. Discuss the impact of debt on the company's financial position."
    + "5. Capital Expenditures: Analyze the company's capital expenditures and investment activities. Comment "
    + "on the company's investment strategy and its implications for future growth."
    + "6. Return on Investment: Calculate and discuss metrics such as return on equity (ROE) and return on "
    + "assets (ROA) to evaluate the company's efficiency in generating returns for shareholders."
}

# Exports
__all__ = [
    "SecDbConfig",
    "SecCliConfig",
    "SecPlotConfig",
    "builtin_prompts",
    "DEFAULT_SEC_DB_CONFIG",
    "DEFAULT_SEC_CLI_CONFIG",
    "DEFAULT_SEC_PLOT_CONFIG",
]

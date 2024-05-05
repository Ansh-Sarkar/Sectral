"""
SecDB.py contains all the necessary classes and functions required to create
or load a vector database with respect to a given stock ticker
"""
import os
import shutil
from tqdm import tqdm
from langchain.vectorstores import FAISS
from sec_edgar_downloader import Downloader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader

from SecConfig import SecDbConfig
from SecConfig import DEFAULT_SEC_DB_CONFIG


class SecDB:
    """
    Class representing a SEC Database.

    Attributes:
        config (SecDbConfig): Configuration for the SEC Database.
    """

    def __init__(self, config: SecDbConfig):
        """Initialize the SecDB object with given configuration."""
        self._config = config
        self._vector_db = None
        self._config.stock_ticker = self._config.stock_ticker.upper()

    def __str__(self):
        """Return string representation of the SecDB object."""
        return "SecDB(" + str(self._config) + ")"

    def __repr__(self):
        """Return string representation of the SecDB object."""
        return "SecDB(" + str(self._config) + ")"

    def _sec_db_print(self, string, end="\n"):
        """Helper function to print messages related to SecDB operations."""
        print("\033[92m" + "SecDB > " + str(string) + "\033[0m", end=end)

    def get_vector_db(self):
        """Get the vector database."""
        return self._vector_db

    def _fetch_data(self):
        """
        Fetch data from SEC Edgar website based on configuration.

        This function downloads filings for the specified company and filing type
        within the specified date range.
        """
        sec_downloader = Downloader(
            self._config.secd_company,
            self._config.secd_email,
            self._config.secd_base_data_dir,
        )
        self._sec_db_print(
            "(Downloading) STOCK TICKER: "
            + self._config.stock_ticker
            + "FILING TYPE: "
            + self._config.filing_type
        )
        sec_downloader.get(
            self._config.filing_type,
            self._config.stock_ticker,
            download_details=True,
            after=self._config.start_date,
            before=self._config.end_date,
        )

    def _get_stock_filing_data_files(self):
        """
        Get paths of stock filing data files.

        Returns:
            list: List of file paths for stock filing data files.
        """
        filings_directory = (
            self._config.secd_base_data_dir
            + "/sec-edgar-filings/"
            + self._config.stock_ticker
            + "/"
            + self._config.filing_type
        )

        stock_filing_dirs = [
            os.path.join(filings_directory, f)
            for f in os.listdir(filings_directory)
            if os.path.isdir(os.path.join(filings_directory, f))
        ]

        stock_filing_data_files = []
        for stock_filing_dir in stock_filing_dirs:
            for stock_data_file in os.listdir(stock_filing_dir):
                stock_data_file_path = os.path.join(stock_filing_dir, stock_data_file)

                if os.path.isfile(stock_data_file_path) and (
                    stock_data_file.endswith(".html")
                    or stock_data_file.endswith(".htm")
                ):
                    stock_filing_data_files.append(stock_data_file_path)

        return stock_filing_data_files

    def _load_existing(self, embeddings):
        """
        Load existing FAISS index if available.

        Args:
            embeddings: Embeddings to use for indexing.
        """
        if os.path.exists(f"./indexstore/{self._config.stock_ticker}/faiss_index"):
            self._sec_db_print(
                f"Found existing FAISS Index for {self._config.stock_ticker}"
            )
            self._sec_db_print(
                f"FORCE_RECALCULATE_INDEX set to : {self._config.force_recalculate_index}"
            )

            # if we do not want the vector db indices to be recalculated then we
            # load the existing index file available locally.
            if not self._config.force_recalculate_index:
                self._vector_db = FAISS.load_local(
                    f"indexstore/{self._config.stock_ticker}/faiss_index",
                    embeddings,
                    # If running in Jupyter Notebook uncomment following line
                    # allow_dangerous_deserialization = True
                )

    def _create_vector_db(self, embeddings, stock_filing_data_files, text_splitter):
        """
        Create vector database from filings data.

        Args:
            embeddings: Embeddings to use for creating vector database.
            stock_filing_data_files (list): List of paths for stock filing data files.
            text_splitter: Text splitter object for splitting documents.

        """
        if self._config.force_recalculate_index or (self._vector_db is None):
            for data_file in tqdm(stock_filing_data_files):
                self._sec_db_print(
                    f"Creating Embeddings & Vector Indices for {data_file}"
                )
                # Loading the HTML file using the Unstructured HTML Loader by LangChain
                # allows efficient structuring of tables and other HTML structures / elements
                loader = UnstructuredHTMLLoader(data_file)
                documents = loader.load()

                # split the document into small chunks
                docs = text_splitter.split_documents(documents)

                # generating the vector database for the document embeddings
                vector_db = FAISS.from_documents(docs, embeddings)
                if self._vector_db is None:
                    self._vector_db = vector_db
                else:
                    self._vector_db.merge_from(vector_db)

                vector_db.save_local(f"indexstore/{data_file.split('.')[0]}_fass_index")

            self._vector_db.save_local(
                f"indexstore/{self._config.stock_ticker}/faiss_index"
            )

    def _cleanup(self):
        """Clean up the data directory."""
        # cleaning up the data directory since it is no longer required
        if os.path.exists(self._config.secd_base_data_dir):
            shutil.rmtree(self._config.secd_base_data_dir)
        self._sec_db_print(f"Cleanup -> {self._config.secd_base_data_dir}")

    def get_database(self):
        """Get the vector database."""
        return self._vector_db

    def init_database(self):
        """Initialize the database."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.model_chunk_size,
            chunk_overlap=self._config.model_chunk_overlap,
        )
        embeddings = HuggingFaceEmbeddings(
            model_name=self._config.embedding_model,
            cache_folder=self._config.embeddings_cache_dir,
        )

        self._load_existing(embeddings)
        if (self._vector_db is not None) and not self._config.force_recalculate_index:
            return self._vector_db

        self._fetch_data()
        stock_filing_data_files = self._get_stock_filing_data_files()

        self._vector_db = None  # FAISS In Memory Loaded (IMP) Vector Database

        self._create_vector_db(embeddings, stock_filing_data_files, text_splitter)
        self._cleanup()

        if self._vector_db is None:
            self._sec_db_print(
                "!! An Error Occurred While Initializing the Vector Database"
            )
        return self._vector_db


if __name__ == "__main__":
    sec_db = SecDB(config=DEFAULT_SEC_DB_CONFIG)
    db = sec_db.init_database()
    print(db.similarity_search("Products & Services"))

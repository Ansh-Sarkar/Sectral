"""
SecCLI.py contains all the necessary classes and functions required to create,
and manage interactive sessions related to the 10-K filings of a given stock ticker
"""
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate

from SecDB import SecDB
from SecPlot import SecPlot
from SecConfig import DEFAULT_SEC_DB_CONFIG
from SecConfig import DEFAULT_SEC_CLI_CONFIG
from SecConfig import DEFAULT_SEC_PLOT_CONFIG
from SecConfig import SecCliConfig, builtin_prompts


class SecCLI:
    """
    Class representing a Command Line Interface for SEC Database operations.

    Attributes:
        config (SecCliConfig): Configuration for the CLI.
        sec_db (SecDB): Instance of the SEC Database.
    """
    def __init__(self, config: SecCliConfig):
        """Initialize the SecCLI object with given configuration."""
        self._config = config
        self._sec_db = None

    def _get_prompt_template(self):
        """Create and return the prompt template."""
        input_template = (
            "<s>"
            + "[INST] Answer the question based only on the following context: [/INST] "
            + "{context}"
            + " </s>"
            + "[INST] Question: {question} [/INST]"
        )

        return PromptTemplate(
            template=input_template, input_variables=["context", "question"]
        )

    def _parse_raw_response(self, raw_response):
        """
        Parse raw response from the model.

        Args:
            raw_response (str): Raw response string.

        Returns:
            str: Parsed response.
        """
        raw_response = raw_response.replace("<s>", "")
        raw_response = raw_response.replace("</s>", "")
        raw_response = raw_response.replace("[INST]", "")
        raw_response = raw_response.replace("[/INST]", "")
        raw_response = raw_response.split("\n")

        final_answer = None
        for idx, val in enumerate(raw_response):
            if "Question:" in val:
                final_answer = raw_response[idx:]

        response = ""
        for idx, _ in enumerate(final_answer):
            if idx == 1:
                response = response + "\n\n[Answer]" + final_answer[idx]
            elif len(final_answer[idx]) > 0:
                response = response + final_answer[idx] + "\n"
        return response

    def _help_menu(self):
        """Generate and return the help menu."""
        return (
            "\n"
            + "\\help : Displays all the available commands.\n"
            + "\\load : Load stock embeddings for Q&A on SEC filings.\n"
            + "\\plots : Plot line graphs for key financial metrics from SEC filings.\n"
            + "\\insights : Gives complete insight + plots (executes built in prompts).\n"
            + "\\chat : Ask the RAG questions regarding loaded stock data.\n"
        )

    def _validate_args(self, command, msg_tokens):
        """Validate the arguments for a given command."""
        command_syntax = "Command Syntax: "
        if command == "load":
            return (len(msg_tokens) == 2, command_syntax + "\\load" + " <STOCK_TICKER>")
        elif command == "plots":
            return (
                len(msg_tokens) == 2,
                command_syntax + "\\plots" + " <STOCK_TICKER>",
            )
        elif command == "chat":
            return (len(msg_tokens) >= 2, command_syntax + "\\chat" + " <PROMPT>")
        elif command == "insights":
            return (len(msg_tokens) == 1, command_syntax + "\\insights")
        elif command == "help":
            return (len(msg_tokens) == 1, command_syntax + "\\help")

    def _sectral_log(self, content):
        """
        Log content with a prefix.

        Args:
            content (str): Content to log.

        Returns:
            str: Logged content.
        """
        return f"""Sectral > {content}"""

    def _chat_handler(self, message):
        """Handle chat messages and generate relevant responses."""
        response = None
        command = message.split()[0]
        msg_tokens = message.split()

        if command == "\\help":
            valid = self._validate_args("help", msg_tokens)
            if valid[0]:
                response = self._sectral_log(self._help_menu())
            else:
                response = valid[1]

        elif command == "\\load":
            valid = self._validate_args("load", msg_tokens)
            if valid[0]:
                try:
                    CUSTOM_DB_CONFIG = DEFAULT_SEC_DB_CONFIG
                    CUSTOM_DB_CONFIG.stock_ticker = msg_tokens[1]
                    self._sec_db = SecDB(config=CUSTOM_DB_CONFIG)
                    _ = self._sec_db.init_database()
                    response = (
                        "Loaded Vector Database & Embeddings for "
                        + CUSTOM_DB_CONFIG.stock_ticker
                    )
                    response = self._sectral_log(response)

                except Exception as e:
                    response = self._sectral_log(f"An Error Occurred: {str(e)}")
            else:
                response = self._sectral_log(valid[1])

        elif command == "\\plots":
            valid = self._validate_args("plots", msg_tokens)
            if valid[0]:
                try:
                    CUSTOM_SEC_PLOT_CONFIG = DEFAULT_SEC_PLOT_CONFIG
                    CUSTOM_SEC_PLOT_CONFIG.stock_ticker = msg_tokens[1]
                    sec_plot = SecPlot(config=CUSTOM_SEC_PLOT_CONFIG)
                    sec_plot.fetch_data()
                    figure_paths = sec_plot.plot_data()

                    response = (
                        f"Plotted a total of {len(figure_paths)} figures on various financial metrics and their "
                        + "progression over the years. Note that all the following figures are w.r.t to the stock ticker "
                        + f"provided : {CUSTOM_SEC_PLOT_CONFIG.stock_ticker}"
                    )

                    for figure_path in figure_paths:
                        response += f"{figure_path}\n"
                    response = self._sectral_log(response)

                except Exception as e:
                    response = self._sectral_log(f"An Error Occurred: {str(e)}")

        elif command == "\\insights":
            valid = self._validate_args("insights", msg_tokens)
            if valid[0]:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self._config.hf_pipe,
                    retriever=self._sec_db.get_vector_db().as_retriever(
                        earch_kwargs={"k": 2}
                    ),  # top 2 results only, speed things up
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": self._get_prompt_template()},
                )
                raw_response = qa_chain.invoke(builtin_prompts["insights"])
                processed_response = self._parse_raw_response(raw_response["result"])
                response = self._sectral_log(processed_response)
            else:
                response = self._sectral_log(valid[1])

        elif command == "\\chat":
            valid = self._validate_args("chat", msg_tokens)
            if valid[0]:
                print("All Good till here . . ..")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self._config.hf_pipe,
                    retriever=self._sec_db.get_vector_db().as_retriever(
                        earch_kwargs={"k": 2}
                    ),  # top 2 results only, speed things up
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": self._get_prompt_template()},
                )
                print("and here")
                raw_response = qa_chain.invoke(" ".join(msg_tokens[1:]))
                print("and here 2")
                processed_response = self._parse_raw_response(raw_response["result"])
                response = self._sectral_log(processed_response)
            else:
                response = self._sectral_log(valid[1])

        else:
            response = f"{command} is not a supported command.\n\n"
            response = self._sectral_log(response + self._help_menu())

        return response

    def launch(self):
        """Launch the command line interface."""
        print("📚 Sectral (Mistral-7B_instruct.v0.1) 🚀")
        while True:
            print("\n               \n")
            message = input("Prompt > ")
            try:
                if message == "\\quit":
                    print("Exiting Interactive Session . . .")
                    break
                else:
                    response = self._chat_handler(message)
                    print(response)
            except Exception as error:
                print(f"An Error Occured While Handling the Request : {error}")


if __name__ == "__main__":
    cli = SecCLI(config=DEFAULT_SEC_CLI_CONFIG)
    cli.launch()

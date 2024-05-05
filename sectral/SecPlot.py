"""
SecPlot.py contains all the necessary classes and functions required to create,
plot and store multiple graphs related to facts / metrics of a given stock ticker
"""
import re
import os
import pandas as pd
import edgar as SEC
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

from SecConfig import SecPlotConfig
from SecConfig import DEFAULT_SEC_PLOT_CONFIG


class SecPlot:
    """
    Class for plotting financial data fetched from SEC filings.
    """

    def __init__(self, config: SecPlotConfig):
        """
        Initializes SecPlot instance.

        Args:
            config (SecPlotConfig): Configuration object for SEC plot.
        """
        self._config = config
        self._categories: list[str] = []
        self._plot_data: dict[str, pd.DataFrame] = {}

    def __str__(self):
        """
        String representation of SecPlot instance.

        Returns:
            str: String representation.
        """
        return "SecPlot(" + str(self._config) + ")"

    def __repr__(self):
        """
        String representation of SecPlot instance.

        Returns:
            str: String representation.
        """
        return "SecPlot(" + str(self._config) + ")"

    def _sec_plot_print(self, string, end="\n"):
        """
        Helper method to print formatted messages related to SecPlot.

        Args:
            string (str): Message to print.
            end (str, optional): Ending character for the print statement. Defaults to "\n".
        """
        print("\033[92m" + "SecPlot > " + str(string) + "\033[0m", end=end)

    def _map_fact_to_label(self, fact_name: str):
        """
        Maps a fact name to a readable label.

        Args:
            fact_name (str): Name of the fact to map.

        Returns:
            str: Readable label for the fact.
        """
        # Split the string based on capital letters
        words = re.findall("[A-Z][^A-Z]*", fact_name)
        words = [word.capitalize() for word in words]
        # Capitalize the first letter of each word and join them with spaces
        readable_name = " ".join(words)
        return readable_name

    def fetch_data(self):
        """
        Fetches financial data from SEC filings.
        """
        SEC.set_identity("A. Sarkar anshsarkar@gmail.com")
        company = SEC.Company(self._config.stock_ticker)
        df = company.get_facts().to_pandas()
        df = df[df[self._config.df_filing] == self._config.filing_type]

        unique_fact_categories = list(df[self._config.df_metric].unique())

        for metric in self._config.fin_metrics:
            if metric in unique_fact_categories:
                self._categories.append(metric)

        for metric in self._categories:
            self._plot_data[metric] = df[df[self._config.df_metric] == metric][
                self._config.df_cols
            ]

    def plot_data(self, save=True, display=False):
        """
        Plots the fetched financial data.

        Args:
            save (bool, optional): Whether to save the plots. Defaults to True.
            display (bool, optional): Whether to display the plots. Defaults to False.

        Returns:
            list: List of paths to saved plot images.
        """
        self._sec_plot_print(
            "The plots may be scaled. Refer to top left corner for scale info"
        )

        sns.set_theme()
        fig_paths = []
        for metric in self._categories:
            record_count = self._plot_data[metric][self._config.df_val].count()
            print(f"Total records for {metric} = {record_count}")
            if record_count < 10:
                print("Skipping Due to Insufficient Data")
            else:
                sns.lineplot(
                    data=self._plot_data[metric],
                    x=self._config.df_x,
                    y=self._config.df_val,
                )
                plt.xlabel(self._config.df_x_label)
                plt.ylabel(self._map_fact_to_label(metric))
                if save:
                    plot_dir = f"{os.getcwd()}/plots/{self._config.stock_ticker}"
                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    fig_save_path = f"{plot_dir}/{str(datetime.now())}.png"
                    plt.savefig(fig_save_path)
                    fig_paths.append(fig_save_path)
                if display:
                    plt.show()
                plt.cla()
                plt.clf()
                plt.close()
        return fig_paths


if __name__ == "__main__":
    sec_plot = SecPlot(config=DEFAULT_SEC_PLOT_CONFIG)
    sec_plot.fetch_data()
    sec_plot.plot_data()

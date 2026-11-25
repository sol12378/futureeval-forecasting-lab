import argparse
import asyncio
import contextvars

import research_sources as research_sources_mod
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Literal

from budget_guard import (
    BudgetExceeded,
    BudgetGuard,
    LedgerError,
    current_month,
    load_last_key_slots,
    load_submitted_keys,
    previous_month,
    question_key,
)
from forecast_config import (
    all_models,
    gap_fill_enabled,
    build_llm_config,
    forecaster_plan,
    research_sources,
    setting,
    config_counts,
    gpt_reasoning_effort,
    has_asknews,
    model_names,
    unpriced_models,
)
from api_keys import KeyPool, configured_key_order

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    ApiFilter,
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = self._get_research_prompt(question, researcher)

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )




_FORECASTER: contextvars.ContextVar[str] = contextvars.ContextVar("forecaster", default="claude")
MC_PROBABILITY_FLOOR = 0.01
GAP_FILL_RESEARCH_CHARS = 40000

# Distribution questions: the template asks for 6 percentiles (10..90). Past top bots asked for
# more points incl. tails; forecasting-tools interpolates the CDF linearly between them and the
# ensemble is aggregated as a point-wise median CDF.
TEMPLATE_PERCENTILES = (0.10, 0.20, 0.40, 0.60, 0.80, 0.90)
WIDE_PERCENTILES = (0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975)
_PERCENTILE_SET: contextvars.ContextVar[tuple[float, ...]] = contextvars.ContextVar(
    "percentile_set", default=TEMPLATE_PERCENTILES)
NUMERIC_BLOCK = (
    "Percentile 10: XX (lowest number value)\nPercentile 20: XX\nPercentile 40: XX\n"
    "Percentile 60: XX\nPercentile 80: XX\nPercentile 90: XX (highest number value)"
)
DATE_BLOCK = (
    "Percentile 10: YYYY-MM-DD (oldest date)\nPercentile 20: YYYY-MM-DD\nPercentile 40: YYYY-MM-DD\n"
    "Percentile 60: YYYY-MM-DD\nPercentile 80: YYYY-MM-DD\nPercentile 90: YYYY-MM-DD (newest date)"
)


def _pct_label(p: float) -> str:
    return f"{p * 100:g}"


def _expand_percentile_block(prompt: str, block: str, fmt: str) -> str | None:
    """Swap the template's 6-line answer format for the 13-point one; None if the template changed."""
    if block not in prompt:
        return None
    lines = []
    for i, p in enumerate(WIDE_PERCENTILES):
        suffix = " (lowest)" if i == 0 else " (highest)" if i == len(WIDE_PERCENTILES) - 1 else ""
        lines.append(f"Percentile {_pct_label(p)}: {fmt}{suffix}")
    note = (
        "\n(The 2.5th and 97.5th percentiles describe genuine tail risk: make them wide enough that "
        "an outcome outside them would truly surprise you.)"
    )
    return prompt.replace(block, "\n".join(lines)) + note


class GuardedBot(FallTemplateBot2026):
    """Template bot + per-provider budget guard + per-question ledger entries.

    The forecasting logic is inherited unchanged from FallTemplateBot2026 so that
    upstream template updates stay easy to merge.
    """

    budget_guard: BudgetGuard | None = None

    def _guard(self, what: str) -> None:
        if self.budget_guard is not None:
            self.budget_guard.assert_can_spend(what)

    def _create_single_distribution_parsing_message(self, question) -> str:
        # The two parser samples must agree, so pin them to exactly the requested percentile set
        # (e.g. Claude sometimes adds an unrequested median line).
        wanted = _PERCENTILE_SET.get()
        listed = ", ".join(_pct_label(p) for p in wanted[:-1]) + f" and {_pct_label(wanted[-1])}"
        return super()._create_single_distribution_parsing_message(question) + (
            f"\n- Return exactly the {len(wanted)} percentiles that were requested: {listed}."
            " Ignore any other percentiles even if they appear in the text."
        )

    async def _distribution_with_tails(self, question, prompt: str, block: str, fmt: str, call):
        """Ask for 13 percentiles (fatter, explicit tails); fall back to the template's 6 on failure."""
        # Discrete questions (cdf_size != 201) failed validation with 13 points in testing: keep 6.
        continuous = getattr(question, "cdf_size", 201) == 201
        expanded = _expand_percentile_block(prompt, block, fmt) if continuous else None
        if expanded is not None:
            token = _PERCENTILE_SET.set(WIDE_PERCENTILES)
            try:
                return await call(question, expanded)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - retry with the proven 6-point prompt
                logger.warning("13-percentile forecast failed for %s (%s); retrying with 6", question.page_url, type(exc).__name__)
                # Details only at INFO: production raises this logger to WARNING (public logs).
                logger.info("13-percentile failure detail: %s", str(exc)[:500])
            finally:
                _PERCENTILE_SET.reset(token)
        return await call(question, prompt)

    async def _numeric_prompt_to_forecast(self, question, prompt):
        return await self._distribution_with_tails(
            question, prompt, NUMERIC_BLOCK, "XX", super()._numeric_prompt_to_forecast)

    async def _date_prompt_to_forecast(self, question, prompt):
        return await self._distribution_with_tails(
            question, prompt, DATE_BLOCK, "YYYY-MM-DD", super()._date_prompt_to_forecast)

    # ------------------------------------------------------------------ research
    research_sources: tuple[str, ...] = ("asknews",)
    gap_fill: bool = False
    forecaster_plan: tuple[str, ...] = ("claude",)

    def _research_semaphore(self) -> asyncio.Semaphore:
        # One semaphore per event loop (main.py runs FutureEval and MiniBench in separate loops).
        loop = asyncio.get_running_loop()
        if getattr(self, "_research_sem_loop", None) is not loop:
            self._research_sem = asyncio.Semaphore(3)
            self._research_sem_loop = loop
        return self._research_sem

    @staticmethod
    def _web_research_prompt(question: MetaculusQuestion) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        closes = getattr(question, "close_time", None)
        resolves = getattr(question, "scheduled_resolution_time", None)
        return clean_indents(
            f"""
            You are the research assistant of a professional superforecaster. Today is {today} (UTC).
            Use web search to gather the most decision-relevant, CURRENT evidence for the question below.
            Do not give a probability yourself.

            Question: {question.question_text}
            Question closes: {closes}; scheduled resolution: {resolves}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Background provided:
            {question.background_info}

            Report concisely (under 600 words), with dates and sources for every fact:
            1. The current status / latest value of whatever the resolution depends on (check the named resolution source if any).
            2. Recent news (most recent first) and anything already scheduled before the resolution date.
            3. Historical base rates or comparable past cases, and recent trend / volatility for numeric quantities.
            4. Expert, official or market expectations (polls, prediction markets, consensus forecasts) if available.
            5. Whether the question may effectively be already decided, and any ambiguity in the resolution criteria.
            """
        )

    async def _research_with(self, source: str, question: MetaculusQuestion) -> str:
        guard = self.budget_guard
        if source == "asknews":
            if guard is not None and not guard.try_reserve_asknews_call():
                logger.warning("AskNews search cap reached; skipping AskNews for %s", question.page_url)
                return ""
            return await super().run_research(question)
        if source == "tavily":
            return await research_sources_mod.tavily(question.question_text)
        if source == "exa":
            return await research_sources_mod.exa(question.question_text)
        if source == "markets":
            return await research_sources_mod.markets(question.question_text)
        if source == "data":
            parser = self.get_llm("parser", "llm")
            return await research_sources_mod.data(question.question_text, question.resolution_criteria, parser.invoke)
        if source == "resolution_pages":
            parser = self.get_llm("parser", "llm")
            return await research_sources_mod.resolution_pages(
                question.question_text, question.resolution_criteria, question.fine_print,
                question.background_info, parser.invoke)
        llm = self.get_llm("claude_researcher" if source == "claude_web" else "gpt_researcher", "llm")
        return await llm.invoke(self._web_research_prompt(question))

    @staticmethod
    def _gap_fill_prompt(question: MetaculusQuestion, research: str) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        closes = getattr(question, "close_time", None)
        resolves = getattr(question, "scheduled_resolution_time", None)
        if len(research) > GAP_FILL_RESEARCH_CHARS:
            research = research[:GAP_FILL_RESEARCH_CHARS] + "\n[... research truncated ...]"
        return clean_indents(
            f"""
            You are the fact-checker of a professional superforecaster. Today is {today} (UTC).
            Below is a forecasting question and the research collected so far by several tools.
            Do not give a probability yourself.

            Question: {question.question_text}
            Question closes: {closes}; scheduled resolution: {resolves}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Research so far:
            <<<
            {research}
            >>>

            Step 1: identify the 2-4 facts that would most change the forecast and are MISSING, STALE
            (older than the latest possible update) or CONFLICTING between sources above. Always include
            the latest value/status of the exact resolution source if it is not clearly established, and
            any event scheduled before the resolution date.
            Step 2: use web search to settle each of them.
            Report under 400 words, one short paragraph per gap: the gap, what you found (with dates and
            sources), and which earlier claim it confirms or corrects. Say plainly when a gap stays open.
            """
        )

    async def _gap_fill(self, question: MetaculusQuestion, research: str) -> str:
        llm = self.get_llm("gpt_researcher", "llm")
        return await llm.invoke(self._gap_fill_prompt(question, research))

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Run every configured source in parallel; a failing source never fails the question."""
        self._guard(f"research for {question.page_url}")
        titles = {
            "claude_web": "Web research (Claude)", "gpt_web": "Web research (GPT)", "asknews": "News (AskNews)",
            "tavily": "News search results (Tavily)", "exa": "Web search results (Exa)",
            "markets": "Prediction markets", "data": "Official data",
            "resolution_pages": "Resolution source pages (read directly)",
        }
        async with self._research_semaphore():
            results = await asyncio.gather(
                *(self._research_with(src, question) for src in self.research_sources), return_exceptions=True
            )
        sections = []
        sizes = {src: (len(r) if isinstance(r, str) else type(r).__name__) for src, r in zip(self.research_sources, results)}
        print(f"Research sizes for {question.page_url}: {sizes}")  # lengths only, safe for public logs
        for source, result in zip(self.research_sources, results):
            if isinstance(result, BudgetExceeded):
                raise result
            if isinstance(result, BaseException):
                # Questions are open only ~3 hours and allow one submission: degrade, don't fail.
                logger.warning("Research source %s failed for %s (%s)", source, question.page_url, type(result).__name__)
                if self.budget_guard is not None:
                    self.budget_guard.note_research_failure()
                continue
            if result and result.strip():
                sections.append(f"## {titles[source]}\n{result.strip()}")
        if self.gap_fill and sections:
            self._guard(f"gap-fill research for {question.page_url}")
            # Second pass: settle the facts the first pass left missing, stale or contradictory.
            try:
                follow_up = await self._gap_fill(question, "\n\n".join(sections))
            except BudgetExceeded:
                raise
            except Exception as exc:  # degrade, don't fail the question
                logger.warning("Gap-fill research failed for %s (%s)", question.page_url, type(exc).__name__)
                follow_up = ""
                if self.budget_guard is not None:
                    self.budget_guard.note_research_failure()
            print(f"Gap-fill research for {question.page_url}: {len(follow_up or '')} chars")
            if follow_up and follow_up.strip():
                sections.append(f"## Follow-up research on gaps and conflicts (GPT)\n{follow_up.strip()}")
        return "\n\n".join(sections)

    # ------------------------------------------------------------- loss guards
    def _get_conditional_disclaimer_if_necessary(self, question: MetaculusQuestion) -> str:
        # Injected into every forecast prompt (binary / MC / numeric / date) by the template.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        closes = getattr(question, "close_time", None)
        resolves = getattr(question, "scheduled_resolution_time", None)
        return super()._get_conditional_disclaimer_if_necessary(question) + "\n" + clean_indents(
            f"""
            Important checks before you answer (today is {today} UTC; question closes {closes}; scheduled resolution {resolves}):
            - Unless the research clearly shows the outcome has ALREADY happened, assume it has not happened yet; this question is still open.
            - "Before <date>" / "by <date>" means the window from now until that date; count how much time is actually left.
            - Base your current values on the research and official data provided, not on memory; if they conflict, prefer the most recent dated source.
            - Re-check units and scale (thousands vs millions, percent vs percentage points) against the question's stated units.
            - Prediction-market prices in the research are only a loose reference: their resolution rules and dates may differ.
            - Historically about two thirds of binary questions in these tournaments resolved No and the status quo usually holds; avoid both overconfidence and reflexive 50%.
            """
        )

    async def _run_forecast_on_multiple_choice(self, question, research):
        # Never leave an option at ~0: one surprise outcome at 0.1% costs far more than the floor.
        prediction = await super()._run_forecast_on_multiple_choice(question, research)
        options = prediction.prediction_value.predicted_options
        floor = min(MC_PROBABILITY_FLOOR, 0.5 / max(len(options), 1))
        raised = [max(option.probability, floor) for option in options]
        total = sum(raised)
        for option, value in zip(options, raised):
            option.probability = value / total
        return prediction

    # ---------------------------------------------------------------- ensemble
    def get_llm(self, purpose="default", guarantee_type=None):
        # Inside a GPT ensemble slot, the template's "default" forecaster resolves to the GPT model.
        if purpose == "default" and _FORECASTER.get() == "gpt":
            purpose = "gpt_forecaster"
        return super().get_llm(purpose, guarantee_type)

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        self._guard(f"prediction for {question.page_url}")
        counters = self.__dict__.setdefault("_prediction_counters", {})
        index = counters.get(question_key(question), 0)
        counters[question_key(question)] = index + 1
        forecaster = self.forecaster_plan[index % len(self.forecaster_plan)]
        token = _FORECASTER.set(forecaster)  # task-local: concurrent predictions don't interfere
        try:
            return await super()._make_prediction(question, research)
        finally:
            _FORECASTER.reset(token)

    async def _run_individual_question(self, question: MetaculusQuestion):
        guard = self.budget_guard
        key = question_key(question)
        try:
            self._guard(f"question {question.page_url}")
        except BudgetExceeded:
            if guard is not None:
                guard.note_question(key, question.page_url, "skipped_budget", "BudgetExceeded")
            raise
        try:
            report = await super()._run_individual_question(question)
        except Exception as exc:
            # The ledger branch is public: record only the exception type, never its message
            # (provider error text could echo request details). Full errors stay in the run log.
            if guard is not None:
                guard.note_question(key, question.page_url, "failed", type(exc).__name__)
            raise
        if guard is not None:
            status = "submitted" if self.publish_reports_to_metaculus else "dry_run"
            guard.note_question(key, question.page_url, status)
        return report


TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}

# Exit codes (non-zero makes the GitHub Actions run fail and email the owner).
EXIT_OK = 0
EXIT_QUESTION_ERRORS = 1
EXIT_CONFIG = 2
EXIT_BUDGET = 3
EXIT_MISSED = 4
EXIT_LEDGER_MISMATCH = 5


def _write_step_summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("```\n" + "\n".join(lines) + "\n```\n")


def _latest_my_forecast_time(question) -> datetime | None:
    history = getattr(question, "previous_forecasts", None) or []
    times = [p.timestamp for p in history if getattr(p, "timestamp", None) is not None]
    return max(times) if times else None


def verify_on_metaculus(client, tournament_id, submitted_keys: set[str], run_started: datetime) -> tuple[list[str], bool]:
    """Re-read the tournament as the bot and confirm each submitted forecast is visible on Metaculus."""
    lines, ok = [], True
    seen = set()
    for q in client.get_all_open_questions_from_tournament(tournament_id):
        key = question_key(q)
        if key not in submitted_keys:
            continue
        seen.add(key)
        latest = _latest_my_forecast_time(q)
        if latest is not None and latest >= run_started:
            lines.append(f"VERIFIED on Metaculus: {q.page_url} (my latest forecast at {latest.isoformat()})")
        elif q.already_forecasted and latest is None:
            lines.append(f"VERIFIED (forecast flag only, no timestamp): {q.page_url}")
        else:
            ok = False
            lines.append(f"NOT VISIBLE on Metaculus: {q.page_url} (latest={latest})")
    for key in sorted(submitted_keys - seen):
        lines.append(f"Could not re-read {key} (question no longer open?)")
    return lines, ok


def find_missed(questions, now: datetime, grace_minutes: int = 60) -> list[str]:
    """Questions that opened at least `grace_minutes` ago and still carry no forecast from us."""
    cutoff = now - timedelta(minutes=grace_minutes)
    missed = []
    for q in questions:
        opened = getattr(q, "open_time", None)
        if opened is None or opened > cutoff:
            continue  # upcoming or too fresh for the 20-minute poller
        if not q.already_forecasted:
            missed.append(str(q.page_url))
    return missed


def audit(hours: int) -> int:
    """Coverage audit: every question opened in the last `hours` must carry our forecast."""
    check_environment(strict=True)
    client = MetaculusClient()
    now = datetime.now(timezone.utc)
    lines, missed = [], []
    for label, tournament_id in (("FutureEval", client.CURRENT_AI_COMPETITION_ID), ("MiniBench", client.CURRENT_MINIBENCH_ID)):
        # Questions that opened before the bot went live are not "missed" (AUDIT_NOT_BEFORE, ISO UTC).
        since = now - timedelta(hours=hours)
        not_before = os.getenv("AUDIT_NOT_BEFORE", "").strip()
        if not_before:
            since = max(since, datetime.fromisoformat(not_before.replace("Z", "+00:00")))
        api_filter = ApiFilter(
            allowed_tournaments=[tournament_id],
            allowed_statuses=["open", "closed", "resolved"],
            open_time_gt=since,
            group_question_mode="unpack_subquestions",
        )
        questions = asyncio.run(client.get_questions_matching_filter(api_filter))
        mine = find_missed(questions, now)
        forecasted = sum(1 for q in questions if q.already_forecasted)
        lines.append(f"{label} ({tournament_id}): opened in last {hours}h={len(questions)}, forecasted={forecasted}, missed={len(mine)}")
        missed += mine
    if missed:
        lines.append("MISSED (opened >60 min ago, no forecast): " + ", ".join(missed))
    else:
        lines.append("Coverage OK: no missed questions.")
    print("\n".join(lines))
    _write_step_summary(lines)
    return EXIT_MISSED if missed else EXIT_OK


SELFCHECK_QUESTIONS = (
    ("Will the US unemployment rate (BLS, series UNRATE) for December 2026 be above 4.5%?",
     "Resolves Yes if the seasonally adjusted U-3 unemployment rate for December 2026 published by BLS exceeds 4.5%."),
    ("Will the price of Bitcoin be above $100,000 on December 31, 2026?",
     "Resolves Yes if the CoinGecko BTC/USD daily close for 2026-12-31 is above $100,000."),
)


def selfcheck() -> int:
    """Health check of the free research sources (no forecasts, no Metaculus writes)."""
    from api_keys import KeyPool

    pools = {p: KeyPool(p) for p in ("anthropic", "openai")}
    for pool in pools.values():
        pool.activate()
    llms = build_llm_config(key_pools=pools)
    parser_llm = llms["parser"]
    sources = [s for s in research_sources() if s in ("tavily", "exa", "markets", "data")]
    ok = True

    async def one(source, text, criteria):
        if source == "data":
            return await research_sources_mod.data(text, criteria, parser_llm.invoke)
        return await getattr(research_sources_mod, source)(text)

    for text, criteria in SELFCHECK_QUESTIONS:
        results = asyncio.run(_gather_sources(one, sources, text, criteria))
        print(f"Q: {text}")
        for source, result in zip(sources, results):
            if isinstance(result, BaseException):
                ok = False
                print(f"  {source}: FAILED ({type(result).__name__})")
            else:
                preview = " ".join(result.split())[:240]
                print(f"  {source}: {len(result)} chars | {preview}")
    return EXIT_OK if ok else EXIT_QUESTION_ERRORS


async def _gather_sources(one, sources, text, criteria):
    return await asyncio.gather(*(one(s, text, criteria) for s in sources), return_exceptions=True)


def run(run_mode: str) -> int:
    check_environment(strict=True)
    live_enabled = os.getenv("ENABLE_LIVE_FORECASTS", "").strip().lower() == "true"
    if run_mode == "metaculus_cup" and os.getenv("ALLOW_METACULUS_CUP", "").lower() != "true":
        print("Metaculus Cup is out of scope for this project (set ALLOW_METACULUS_CUP=true to override).")
        return EXIT_OK
    if run_mode in ("tournament", "metaculus_cup") and not live_enabled:
        print("ENABLE_LIVE_FORECASTS is not 'true': live tournament submission is disabled. No LLM calls made.")
        return EXIT_OK

    claude_model, gpt_model = model_names()
    try:
        missing_prices = unpriced_models(all_models())
    except ValueError as exc:
        print(f"CONFIG_ERROR: {exc}")
        return EXIT_CONFIG
    if missing_prices:
        print(f"CONFIG_ERROR: litellm cannot price {missing_prices}; budget guard would be blind. Refusing to run.")
        return EXIT_CONFIG
    try:
        guard = BudgetGuard.from_env(run_mode)
        research_count, _unused = config_counts()
        plan = forecaster_plan()  # one prediction per entry, median-aggregated
        sources = research_sources()
        gap_fill = gap_fill_enabled()
        gpt_effort = gpt_reasoning_effort()
        # Start on the key slots the previous run this month ended on (slot 1 at month start).
        # An explicit CLAUDE_KEY_ORDER / GPT_KEY_ORDER always starts on its first key: earlier runs
        # may have used another order, and an empty key costs only one rejected call to skip.
        start_slots = load_last_key_slots(guard.ledger_dir, guard.month)
        key_pools = {}
        for p in ("anthropic", "openai"):
            order = configured_key_order(p)
            key_pools[p] = KeyPool(p, start_slot=order[0] if order else start_slots[p], order=order)
    except (LedgerError, ValueError) as exc:
        print(f"BUDGET_CONFIG_ERROR: {exc}")
        return EXIT_CONFIG
    # Budgets are per key: attaching the pools lets the guard count keys and rotate keys early.
    guard.attach_key_pools(key_pools)

    print("\n".join(guard.summary_lines()))
    level = guard.level()
    if level == "hard":
        print("BUDGET_HARD_STOP: monthly budget threshold reached; not starting any LLM work.")
        guard.write_record({"refused": "budget_hard_stop"})
        _write_step_summary(guard.summary_lines() + ["BUDGET_HARD_STOP"])
        return EXIT_BUDGET
    if level == "soft" and len(plan) > 1:
        print(f"BUDGET_SOFT_LIMIT: reducing the ensemble {plan} -> {plan[:1]}")
        plan = plan[:1]
    prediction_count = len(plan)
    for pool in key_pools.values():
        pool.activate()
    guard.install()

    if run_mode != "test_questions" and os.getenv("LOG_FORECAST_DETAILS", "").lower() != "true":
        # Actions logs of a public repo are public; keep open-question reasoning/predictions out of them.
        # The reasoning is still posted to Metaculus as the bot's (private) comment.
        for name in ("__main__", "forecasting_tools"):
            logging.getLogger(name).setLevel(logging.WARNING)

    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)
    print(
        f"Forecasters: {plan} (claude={claude_model}, gpt={setting(os.environ, 'GPT_FORECAST_MODEL')}), "
        f"parser={gpt_model} (reasoning={gpt_effort}), research={sources}, "
        f"keys: Claude {key_pools['anthropic'].size} (slot {key_pools['anthropic'].slot}), "
        f"GPT {key_pools['openai'].size} (slot {key_pools['openai'].slot}), "
        f"research_reports={research_count}, predictions_per_report={prediction_count}"
    )

    bot = GuardedBot(
        research_reports_per_question=research_count,
        predictions_per_research_report=prediction_count,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        # The summary is not used for forecasting (use_research_summary_to_forecast=False); skip its cost.
        enable_summarize_research=False,
        llms=build_llm_config(key_pools=key_pools),
    )
    bot.budget_guard = guard
    bot.forecaster_plan = tuple(plan)
    bot.research_sources = tuple(sources)
    bot.gap_fill = gap_fill

    client = MetaculusClient()
    forecast_reports: list = []
    exit_code = EXIT_OK
    notes: list[str] = []
    try:
        if run_mode == "tournament":
            submitted = load_submitted_keys(guard.ledger_dir, [guard.month, previous_month(guard.month)])
            targets = [
                ("FutureEval", client.CURRENT_AI_COMPETITION_ID),
                ("MiniBench", client.CURRENT_MINIBENCH_ID),
            ]
            expected: dict[str, str] = {}
            for label, tournament_id in targets:
                questions = client.get_all_open_questions_from_tournament(tournament_id)
                done = [q for q in questions if q.already_forecasted]
                blocked = [q for q in questions if not q.already_forecasted and question_key(q) in submitted]
                fresh = [q for q in questions if not q.already_forecasted and question_key(q) not in submitted]
                notes.append(
                    f"{label} ({tournament_id}): open={len(questions)}, already_forecasted={len(done)}, "
                    f"to_forecast={len(fresh)}, ledger_blocked={len(blocked)}"
                )
                if blocked:
                    # Ledger says we submitted but Metaculus shows no forecast: never risk a duplicate.
                    notes.append("LEDGER_MISMATCH: " + ", ".join(str(q.page_url) for q in blocked))
                    exit_code = max(exit_code, EXIT_LEDGER_MISMATCH)
                for q in fresh:
                    expected[question_key(q)] = str(q.page_url)
                if fresh:
                    forecast_reports += asyncio.run(bot.forecast_questions(fresh, return_exceptions=True))
            if expected:
                still_missing = []
                for _label, tournament_id in targets:
                    for q in client.get_all_open_questions_from_tournament(tournament_id):
                        if question_key(q) in expected and not q.already_forecasted:
                            still_missing.append(str(q.page_url))
                if still_missing:
                    notes.append("MISSED_SUBMISSIONS (still open, no forecast): " + ", ".join(still_missing))
                    exit_code = max(exit_code, EXIT_MISSED)
                else:
                    notes.append(f"Post-check OK: all {len(expected)} targeted question(s) now show a forecast.")
            else:
                notes.append("No new questions to forecast (monitoring OK).")
        elif run_mode == "test_questions":
            # Bot Testing Area allows resubmission; always forecast every open test question.
            bot.skip_previously_forecasted_questions = False
            questions = client.get_all_open_questions_from_tournament("bot-testing-area")
            notes.append(f"Bot Testing Area: open={len(questions)}")
            run_started = datetime.now(timezone.utc)
            forecast_reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
            submitted = {k for k, v in guard.questions.items() if v["status"] == "submitted"}
            if submitted:
                lines, ok = verify_on_metaculus(client, "bot-testing-area", submitted, run_started)
                notes.extend(lines)
                if not ok:
                    exit_code = max(exit_code, EXIT_MISSED)
        elif run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            forecast_reports = asyncio.run(
                bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)
            )
    finally:
        statuses = [q["status"] for q in guard.questions.values()]
        if any(s == "skipped_budget" for s in statuses):
            exit_code = max(exit_code, EXIT_BUDGET)
        idle = run_mode == "tournament" and not guard.questions and guard.run_total() == 0 and exit_code == EXIT_OK
        record_path = "(idle run: no record written)" if idle else guard.write_record(
            {
                "models": {"forecaster": claude_model, "parser": gpt_model},
                "keys_switched_from": {p: pool.switched_from for p, pool in key_pools.items()},
                "asknews": has_asknews(),
                "predictions_per_report": prediction_count,
                "forecaster_plan": plan,
                "research_sources": sources,
                "notes": notes,
            }
        )
        lines = notes + guard.summary_lines() + [f"Ledger record: {record_path}"]
        print("\n".join(lines))
        _write_step_summary(lines)

    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
    if any(isinstance(r, BaseException) for r in forecast_reports):
        exit_code = max(exit_code, EXIT_QUESTION_ERRORS)
    if run_mode == "test_questions":
        bot.log_report_summary(forecast_reports, raise_errors=False)
    return exit_code


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the FutureEval forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions", "audit", "selfcheck"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    parser.add_argument("--hours", type=int, default=24, help="audit window in hours (audit mode)")
    args = parser.parse_args()
    if args.mode == "audit":
        sys.exit(audit(args.hours))
    if args.mode == "selfcheck":
        check_environment(strict=True)
        sys.exit(selfcheck())
    sys.exit(run(args.mode))

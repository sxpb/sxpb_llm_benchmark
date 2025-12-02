from typing import (
    List,
    Dict,
    Any,
    TypedDict,
    Literal,
    Generator,
    Callable,
    Optional,
    cast,
)
from itertools import count
from src.bench.toon.datasets import (
    Employee,
    Order,
    AnalyticsMetric,
    Repository,
    EventLog,
    NestedConfig,
    Dataset,
)

QuestionType = Literal[
    "field-retrieval",
    "retrieval",
    "aggregation",
    "filtering",
    "structure-awareness",
    "structural-validation",
]

AnswerType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "csv-list-ordered",
    "csv-list-unordered",
]


class NormalizationOptions(TypedDict, total=False):
    case_sensitive: bool
    decimal_places: int


class Question(TypedDict):
    id: str
    prompt: str
    groundTruth: str
    type: QuestionType
    dataset: str
    answerType: AnswerType
    normalizationOptions: Optional[NormalizationOptions]


class TabularThresholds(TypedDict):
    salaryRanges: List[int]
    experienceYears: List[int]
    departmentSalaryThreshold: int
    departmentExperienceThreshold: int


class NestedThresholds(TypedDict):
    highValueOrders: List[int]
    statusValueThreshold: int
    itemCountThreshold: int
    totalThresholdsForItems: List[int]


class AnalyticsThresholds(TypedDict):
    views: List[int]
    conversions: List[int]
    viewsForFiltering: List[int]
    conversionsForFiltering: int
    revenueThresholds: List[int]
    viewsThresholdForRevenue: int
    clicksForFiltering: List[int]
    conversionsForClickFiltering: int
    revenueForBounceRate: List[int]
    bounceRateThreshold: float


class StarForkCombination(TypedDict):
    stars: int
    forks: int


class StarWatcherCombination(TypedDict):
    stars: int
    watchers: int


class GithubThresholds(TypedDict):
    stars: List[int]
    forks: List[int]
    watchers: List[int]
    starForkCombinations: List[StarForkCombination]
    starWatcherCombinations: List[StarWatcherCombination]


class QuestionThresholds(TypedDict):
    tabular: TabularThresholds
    nested: NestedThresholds
    analytics: AnalyticsThresholds
    github: GithubThresholds


QUESTION_THRESHOLDS: QuestionThresholds = {
    "tabular": {
        "salaryRanges": [60000, 80000, 100000],
        "experienceYears": [5, 10, 15, 20],
        "departmentSalaryThreshold": 80000,
        "departmentExperienceThreshold": 10,
    },
    "nested": {
        "highValueOrders": [200, 400, 600],
        "statusValueThreshold": 300,
        "itemCountThreshold": 3,
        "totalThresholdsForItems": [300, 500],
    },
    "analytics": {
        "views": [6000],
        "conversions": [20],
        "viewsForFiltering": [6000, 7000],
        "conversionsForFiltering": 15,
        "revenueThresholds": [1000, 1500, 2000],
        "viewsThresholdForRevenue": 6000,
        "clicksForFiltering": [250, 400],
        "conversionsForClickFiltering": 15,
        "revenueForBounceRate": [1000, 1500],
        "bounceRateThreshold": 0.5,
    },
    "github": {
        "stars": [100000, 150000, 200000],
        "forks": [20000, 35000],
        "watchers": [8000],
        "starForkCombinations": [
            {"stars": 75000, "forks": 15000},
            {"stars": 100000, "forks": 20000},
            {"stars": 150000, "forks": 30000},
            {"stars": 200000, "forks": 45000},
        ],
        "starWatcherCombinations": [
            {"stars": 100000, "watchers": 7000},
            {"stars": 150000, "watchers": 9000},
        ],
    },
}

QUESTION_LIMITS = {
    "tabular": {
        "fieldRetrieval": 12,
        "aggregationDepartments": 3,
        "filteringMultiConditionDepartments": 5,
        "filteringExperience": 3,
        "filteringDepartmentExp": 3,
        "filteringDepartmentActive": 2,
    },
    "nested": {
        "fieldRetrievalOrders": 8,
        "fieldRetrievalCustomers": 8,
        "aggregationStatuses": 3,
        "filteringStatusAndValue": 4,
        "filteringStatusAndItems": 3,
    },
    "analytics": {
        "fieldRetrievalDates": 9,
    },
    "github": {
        "fieldRetrievalRepos": 11,
        "aggregationBranches": 2,
        "filteringStarsAndForks": 3,
    },
    "eventLogs": {
        "fieldRetrieval": 10,
        "aggregationEndpoints": 2,
        "filteringLevelAndStatus": 3,
        "filteringEndpointAndStatus": 3,
        "filteringEndpointRetryable": 2,
    },
    "nestedConfig": {
        "fieldRetrieval": 10,
        "filteringComplex": 5,
    },
}

SAMPLE_STRIDES = {
    "EMPLOYEE_FIELD": 2,
    "ORDER_FIELD": 2,
    "CUSTOMER_FIELD": 2,
    "ANALYTICS_FIELD": 3,
    "METRIC_FIELD": 3,
    "REPO_FIELD": 7,
    "EVENT_LOG_FIELD": 5,
}


def create_id_generator() -> Generator[str, None, None]:
    for i in count(1):
        yield f"q{i}"


class QuestionBuilder:
    def __init__(self):
        self._question: Dict[str, Any] = {}

    def id(self, q_id: str) -> "QuestionBuilder":
        self._question["id"] = q_id
        return self

    def prompt(self, prompt: str) -> "QuestionBuilder":
        self._question["prompt"] = prompt
        return self

    def ground_truth(self, ground_truth: str) -> "QuestionBuilder":
        self._question["groundTruth"] = ground_truth
        return self

    def type(self, q_type: QuestionType) -> "QuestionBuilder":
        self._question["type"] = q_type
        return self

    def dataset(self, dataset: str) -> "QuestionBuilder":
        self._question["dataset"] = dataset
        return self

    def answer_type(self, answer_type: AnswerType) -> "QuestionBuilder":
        self._question["answerType"] = answer_type
        return self

    def normalize(self, options: NormalizationOptions) -> "QuestionBuilder":
        self._question["normalizationOptions"] = options
        return self

    def build(self) -> Question:
        if not all(
            k in self._question
            for k in ["id", "prompt", "groundTruth", "type", "dataset"]
        ):
            raise ValueError("Incomplete question")
        return cast(Question, self._question)


def rotate_questions(
    items: List[Any],
    generators: List[Callable],
    limit: int,
    stride: int,
    get_id: Callable,
) -> List[Question]:
    questions = []
    for i in range(min(limit, len(items))):
        item_index = i * stride
        item = items[item_index] if item_index < len(items) else items[i]
        if not item:
            continue
        generator = generators[i % len(generators)]
        if generator:
            questions.append(generator(item, get_id))
    return questions


def generate_analytics_questions(
    metrics: List[AnalyticsMetric], get_id: Callable
) -> List[Question]:
    questions: List[Question] = []

    metric_field_generators = [
        lambda metric, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What are the views for {metric['date']}?")
        .ground_truth(str(metric["views"]))
        .type("field-retrieval")
        .dataset("analytics")
        .answer_type("integer")
        .build(),
        lambda metric, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the revenue for {metric['date']}?")
        .ground_truth(f"{metric['revenue']:.2f}")
        .type("field-retrieval")
        .dataset("analytics")
        .answer_type("number")
        .normalize({"decimal_places": 2})
        .build(),
        lambda metric, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the bounce rate for {metric['date']}?")
        .ground_truth(f"{metric['bounceRate']:.2f}")
        .type("field-retrieval")
        .dataset("analytics")
        .answer_type("number")
        .normalize({"decimal_places": 2})
        .build(),
        lambda metric, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"How many conversions were there on {metric['date']}?")
        .ground_truth(str(metric["conversions"]))
        .type("field-retrieval")
        .dataset("analytics")
        .answer_type("integer")
        .build(),
    ]
    questions.extend(
        rotate_questions(
            metrics,
            metric_field_generators,
            QUESTION_LIMITS["analytics"]["fieldRetrievalDates"],
            SAMPLE_STRIDES["ANALYTICS_FIELD"],
            get_id,
        )
    )

    total_days = len(metrics)
    total_views = sum(m["views"] for m in metrics)
    total_conversions = sum(m["conversions"] for m in metrics)
    total_revenue = sum(m["revenue"] for m in metrics)
    avg_bounce_rate = (
        sum(m["bounceRate"] for m in metrics) / total_days if total_days > 0 else 0
    )

    questions.extend(
        [
            QuestionBuilder()
            .id(get_id())
            .prompt("How many days of data are in the dataset?")
            .ground_truth(str(total_days))
            .type("aggregation")
            .dataset("analytics")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the total number of views across all dates?")
            .ground_truth(str(total_views))
            .type("aggregation")
            .dataset("analytics")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the total number of conversions across all dates?")
            .ground_truth(str(total_conversions))
            .type("aggregation")
            .dataset("analytics")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the total revenue across all dates?")
            .ground_truth(f"{total_revenue:.2f}")
            .type("aggregation")
            .dataset("analytics")
            .answer_type("number")
            .normalize({"decimal_places": 2})
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the average bounce rate?")
            .ground_truth(f"{avg_bounce_rate:.2f}")
            .type("aggregation")
            .dataset("analytics")
            .answer_type("number")
            .normalize({"decimal_places": 2})
            .build(),
        ]
    )

    for threshold in QUESTION_THRESHOLDS["analytics"]["views"]:
        count = sum(1 for m in metrics if m["views"] > threshold)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many days had more than {threshold} views?")
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("analytics")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["analytics"]["conversions"]:
        count = sum(1 for m in metrics if m["conversions"] > threshold)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many days had more than {threshold} conversions?")
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("analytics")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["analytics"]["viewsForFiltering"]:
        count = sum(
            1
            for m in metrics
            if m["views"] > threshold
            and m["conversions"]
            > QUESTION_THRESHOLDS["analytics"]["conversionsForFiltering"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many days had more than {threshold} views and more than {QUESTION_THRESHOLDS['analytics']['conversionsForFiltering']} conversions?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("analytics")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["analytics"]["revenueThresholds"]:
        count = sum(
            1
            for m in metrics
            if m["revenue"] > threshold
            and m["views"]
            > QUESTION_THRESHOLDS["analytics"]["viewsThresholdForRevenue"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many days had revenue greater than {threshold} with views above {QUESTION_THRESHOLDS['analytics']['viewsThresholdForRevenue']}?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("analytics")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["analytics"]["clicksForFiltering"]:
        count = sum(
            1
            for m in metrics
            if m["clicks"] > threshold
            and m["conversions"]
            > QUESTION_THRESHOLDS["analytics"]["conversionsForClickFiltering"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many days had more than {threshold} clicks and more than {QUESTION_THRESHOLDS['analytics']['conversionsForClickFiltering']} conversions?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("analytics")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["analytics"]["revenueForBounceRate"]:
        count = sum(
            1
            for m in metrics
            if m["revenue"] > threshold
            and m["bounceRate"]
            < QUESTION_THRESHOLDS["analytics"]["bounceRateThreshold"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many days had revenue greater than {threshold} with bounce rate below {QUESTION_THRESHOLDS['analytics']['bounceRateThreshold']}?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("analytics")
            .answer_type("integer")
            .build()
        )

    return questions


def generate_event_logs_questions(
    logs: List[EventLog], get_id: Callable
) -> List[Question]:
    questions: List[Question] = []

    log_field_generators = [
        lambda log, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the level of the log at {log['timestamp']}?")
        .ground_truth(log["level"])
        .type("field-retrieval")
        .dataset("event-logs")
        .answer_type("string")
        .build(),
        lambda log, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the endpoint for the log at {log['timestamp']}?")
        .ground_truth(log["endpoint"])
        .type("field-retrieval")
        .dataset("event-logs")
        .answer_type("string")
        .build(),
        lambda log, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the status code for the log at {log['timestamp']}?")
        .ground_truth(str(log["statusCode"]))
        .type("field-retrieval")
        .dataset("event-logs")
        .answer_type("integer")
        .build(),
        lambda log, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the response time for the log at {log['timestamp']}?")
        .ground_truth(str(log["responseTime"]))
        .type("field-retrieval")
        .dataset("event-logs")
        .answer_type("integer")
        .build(),
    ]
    questions.extend(
        rotate_questions(
            logs,
            log_field_generators,
            QUESTION_LIMITS["eventLogs"]["fieldRetrieval"],
            SAMPLE_STRIDES["EVENT_LOG_FIELD"],
            get_id,
        )
    )

    total_logs = len(logs)
    avg_response_time = (
        sum(log["responseTime"] for log in logs) / total_logs if total_logs > 0 else 0
    )

    questions.extend(
        [
            QuestionBuilder()
            .id(get_id())
            .prompt("How many log entries are in the dataset?")
            .ground_truth(str(total_logs))
            .type("aggregation")
            .dataset("event-logs")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the average response time across all logs?")
            .ground_truth(f"{avg_response_time:.2f}")
            .type("aggregation")
            .dataset("event-logs")
            .answer_type("number")
            .normalize({"decimal_places": 2})
            .build(),
        ]
    )

    levels = sorted(list(set(log["level"] for log in logs)))
    for level in levels:
        count = sum(1 for log in logs if log["level"] == level)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f'How many log entries have level "{level}"?')
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("event-logs")
            .answer_type("integer")
            .build()
        )

    endpoints = sorted(list(set(log["endpoint"] for log in logs)))
    for endpoint in endpoints[: QUESTION_LIMITS["eventLogs"]["aggregationEndpoints"]]:
        count = sum(1 for log in logs if log["endpoint"] == endpoint)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f'How many log entries are for endpoint "{endpoint}"?')
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("event-logs")
            .answer_type("integer")
            .build()
        )

    error_count = sum(1 for log in logs if log["statusCode"] >= 400)
    success_count = sum(1 for log in logs if 200 <= log["statusCode"] < 300)

    questions.extend(
        [
            QuestionBuilder()
            .id(get_id())
            .prompt(
                "How many log entries have a status code indicating an error (>= 400)?"
            )
            .ground_truth(str(error_count))
            .type("aggregation")
            .dataset("event-logs")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many log entries have a successful status code (200-299)?")
            .ground_truth(str(success_count))
            .type("aggregation")
            .dataset("event-logs")
            .answer_type("integer")
            .build(),
        ]
    )

    retryable_error_count = sum(
        1 for log in logs if log.get("error", {}).get("retryable")
    )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many log entries have a retryable error?")
        .ground_truth(str(retryable_error_count))
        .type("aggregation")
        .dataset("event-logs")
        .answer_type("integer")
        .build()
    )

    for level in levels[: QUESTION_LIMITS["eventLogs"]["filteringLevelAndStatus"]]:
        if level == "info":
            continue
        count = sum(
            1 for log in logs if log["level"] == level and log["statusCode"] >= 400
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f'How many log entries have level "{level}" and status code >= 400?'
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("event-logs")
            .answer_type("integer")
            .build()
        )

    for endpoint in endpoints[
        : QUESTION_LIMITS["eventLogs"]["filteringEndpointAndStatus"]
    ]:
        count = sum(
            1
            for log in logs
            if log["endpoint"] == endpoint and log["statusCode"] >= 500
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f'How many log entries are for endpoint "{endpoint}" with status code >= 500?'
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("event-logs")
            .answer_type("integer")
            .build()
        )

    for endpoint in endpoints[
        : QUESTION_LIMITS["eventLogs"]["filteringEndpointRetryable"]
    ]:
        count = sum(
            1
            for log in logs
            if log["endpoint"] == endpoint and log.get("error", {}).get("retryable")
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f'How many log entries for endpoint "{endpoint}" have a retryable error?'
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("event-logs")
            .answer_type("integer")
            .build()
        )

    return questions


def generate_github_questions(
    repos: List[Repository], get_id: Callable
) -> List[Question]:
    questions: List[Question] = []

    repo_field_generators = [
        lambda repo, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"How many stars does {repo['repo']} have?")
        .ground_truth(str(repo["stars"]))
        .type("field-retrieval")
        .dataset("github")
        .answer_type("integer")
        .build(),
        lambda repo, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"How many forks does {repo['repo']} have?")
        .ground_truth(str(repo["forks"]))
        .type("field-retrieval")
        .dataset("github")
        .answer_type("integer")
        .build(),
        lambda repo, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"How many watchers does {repo['repo']} have?")
        .ground_truth(str(repo["watchers"]))
        .type("field-retrieval")
        .dataset("github")
        .answer_type("integer")
        .build(),
        lambda repo, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the main branch of {repo['repo']}?")
        .ground_truth(repo["defaultBranch"])
        .type("field-retrieval")
        .dataset("github")
        .answer_type("string")
        .normalize({"case_sensitive": True})
        .build(),
    ]
    questions.extend(
        rotate_questions(
            repos,
            repo_field_generators,
            QUESTION_LIMITS["github"]["fieldRetrievalRepos"],
            SAMPLE_STRIDES["REPO_FIELD"],
            get_id,
        )
    )

    total_repos = len(repos)
    total_stars = sum(r["stars"] for r in repos)
    total_forks = sum(r["forks"] for r in repos)
    avg_stars = total_stars / total_repos if total_repos > 0 else 0

    questions.extend(
        [
            QuestionBuilder()
            .id(get_id())
            .prompt("How many repositories are in the dataset?")
            .ground_truth(str(total_repos))
            .type("aggregation")
            .dataset("github")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the total number of stars across all repositories?")
            .ground_truth(str(total_stars))
            .type("aggregation")
            .dataset("github")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the total number of forks across all repositories?")
            .ground_truth(str(total_forks))
            .type("aggregation")
            .dataset("github")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the average number of stars per repository?")
            .ground_truth(str(round(avg_stars)))
            .type("aggregation")
            .dataset("github")
            .answer_type("integer")
            .build(),
        ]
    )

    branches = sorted(list(set(r["defaultBranch"] for r in repos)))
    for branch in branches[: QUESTION_LIMITS["github"]["aggregationBranches"]]:
        count = sum(1 for r in repos if r["defaultBranch"] == branch)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f'How many repositories use "{branch}" as their default branch?')
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("github")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["github"]["stars"]:
        count = sum(1 for r in repos if r["stars"] > threshold)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many repositories have more than {threshold} stars?")
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("github")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["github"]["forks"]:
        count = sum(1 for r in repos if r["forks"] > threshold)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many repositories have more than {threshold} forks?")
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("github")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["github"]["watchers"]:
        count = sum(1 for r in repos if r["watchers"] > threshold)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many repositories have more than {threshold} watchers?")
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("github")
            .answer_type("integer")
            .build()
        )

    for combo in QUESTION_THRESHOLDS["github"]["starForkCombinations"][
        : QUESTION_LIMITS["github"]["filteringStarsAndForks"]
    ]:
        count = sum(
            1
            for r in repos
            if r["stars"] > combo["stars"] and r["forks"] > combo["forks"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many repositories have more than {combo['stars']} stars and more than {combo['forks']} forks?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("github")
            .answer_type("integer")
            .build()
        )

    for combo in QUESTION_THRESHOLDS["github"]["starWatcherCombinations"]:
        count = sum(
            1
            for r in repos
            if r["stars"] > combo["stars"] and r["watchers"] > combo["watchers"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many repositories have more than {combo['stars']} stars and more than {combo['watchers']} watchers?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("github")
            .answer_type("integer")
            .build()
        )

    return questions


def generate_nested_questions(orders: List[Order], get_id: Callable) -> List[Question]:
    questions: List[Question] = []

    order_field_generators = [
        lambda order, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the total for order {order['orderId']}?")
        .ground_truth(f"{order['total']:.2f}")
        .type("field-retrieval")
        .dataset("nested")
        .answer_type("number")
        .normalize({"decimal_places": 2})
        .build(),
        lambda order, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the status of order {order['orderId']}?")
        .ground_truth(order["status"])
        .type("field-retrieval")
        .dataset("nested")
        .answer_type("string")
        .build(),
    ]
    questions.extend(
        rotate_questions(
            orders,
            order_field_generators,
            QUESTION_LIMITS["nested"]["fieldRetrievalOrders"],
            SAMPLE_STRIDES["ORDER_FIELD"],
            get_id,
        )
    )

    customer_field_generators = [
        lambda order, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the customer name for order {order['orderId']}?")
        .ground_truth(order["customer"]["name"])
        .type("field-retrieval")
        .dataset("nested")
        .answer_type("string")
        .build(),
        lambda order, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the customer email for order {order['orderId']}?")
        .ground_truth(order["customer"]["email"])
        .type("field-retrieval")
        .dataset("nested")
        .answer_type("string")
        .build(),
        lambda order, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the order date for order {order['orderId']}?")
        .ground_truth(order["orderDate"] or "")
        .type("field-retrieval")
        .dataset("nested")
        .answer_type("string")
        .build(),
        lambda order, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"How many items are in order {order['orderId']}?")
        .ground_truth(str(len(order["items"])))
        .type("field-retrieval")
        .dataset("nested")
        .answer_type("integer")
        .build(),
    ]
    customer_orders = [
        orders[i * SAMPLE_STRIDES["CUSTOMER_FIELD"] + 1]
        if i * SAMPLE_STRIDES["CUSTOMER_FIELD"] + 1 < len(orders)
        else orders[i]
        for i in range(len(orders))
    ]
    questions.extend(
        rotate_questions(
            customer_orders,
            customer_field_generators,
            QUESTION_LIMITS["nested"]["fieldRetrievalCustomers"],
            1,
            get_id,
        )
    )

    total_revenue = sum(o["total"] for o in orders)
    avg_order_value = total_revenue / len(orders) if orders else 0
    total_orders = len(orders)
    max_order_value = max(o["total"] for o in orders) if orders else 0

    statuses = sorted(list(set(o["status"] for o in orders)))
    for status in statuses[: QUESTION_LIMITS["nested"]["aggregationStatuses"]]:
        count = sum(1 for o in orders if o["status"] == status)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f'How many orders have status "{status}"?')
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("nested")
            .answer_type("integer")
            .build()
        )

    questions.extend(
        [
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the total revenue across all orders?")
            .ground_truth(f"{total_revenue:.2f}")
            .type("aggregation")
            .dataset("nested")
            .answer_type("number")
            .normalize({"decimal_places": 2})
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the average order value?")
            .ground_truth(f"{avg_order_value:.2f}")
            .type("aggregation")
            .dataset("nested")
            .answer_type("number")
            .normalize({"decimal_places": 2})
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many orders are in the dataset?")
            .ground_truth(str(total_orders))
            .type("aggregation")
            .dataset("nested")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the highest order total?")
            .ground_truth(f"{max_order_value:.2f}")
            .type("aggregation")
            .dataset("nested")
            .answer_type("number")
            .normalize({"decimal_places": 2})
            .build(),
        ]
    )

    for threshold in QUESTION_THRESHOLDS["nested"]["highValueOrders"]:
        count = sum(1 for o in orders if o["total"] > threshold)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many orders have a total greater than {threshold}?")
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("nested")
            .answer_type("integer")
            .build()
        )

    for status in statuses[: QUESTION_LIMITS["nested"]["filteringStatusAndValue"]]:
        count = sum(
            1
            for o in orders
            if o["status"] == status
            and o["total"] > QUESTION_THRESHOLDS["nested"]["statusValueThreshold"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f'How many orders have status "{status}" and total greater than {QUESTION_THRESHOLDS["nested"]["statusValueThreshold"]}?'
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("nested")
            .answer_type("integer")
            .build()
        )

    for status in statuses[: QUESTION_LIMITS["nested"]["filteringStatusAndItems"]]:
        count = sum(
            1
            for o in orders
            if o["status"] == status
            and len(o["items"]) >= QUESTION_THRESHOLDS["nested"]["itemCountThreshold"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f'How many orders have status "{status}" and at least {QUESTION_THRESHOLDS["nested"]["itemCountThreshold"]} items?'
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("nested")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["nested"]["totalThresholdsForItems"]:
        count = sum(
            1
            for o in orders
            if o["total"] > threshold
            and len(o["items"]) >= QUESTION_THRESHOLDS["nested"]["itemCountThreshold"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many orders have a total greater than {threshold} and at least {QUESTION_THRESHOLDS['nested']['itemCountThreshold']} items?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("nested")
            .answer_type("integer")
            .build()
        )

    return questions


def generate_nested_config_questions(
    config: Optional[NestedConfig], get_id: Callable
) -> List[Question]:
    questions: List[Question] = []
    if not config:
        return questions

    field_retrieval_questions = [
        {
            "prompt": "What is the environment in the configuration?",
            "groundTruth": config["environment"],
            "answerType": "string",
        },
        {
            "prompt": "What is the database host?",
            "groundTruth": config["database"]["host"],
            "answerType": "string",
        },
        {
            "prompt": "What is the database port?",
            "groundTruth": str(config["database"]["port"]),
            "answerType": "integer",
        },
        {
            "prompt": "What is the maximum connection pool size?",
            "groundTruth": str(config["database"]["pool"]["max"]),
            "answerType": "integer",
        },
        {
            "prompt": "What is the session duration?",
            "groundTruth": str(config["authentication"]["session"]["duration"]),
            "answerType": "integer",
        },
        {
            "prompt": "What is the minimum connection pool size?",
            "groundTruth": str(config["database"]["pool"]["min"]),
            "answerType": "integer",
        },
        {
            "prompt": "What is the connection pool idle timeout?",
            "groundTruth": str(config["database"]["pool"]["idleTimeout"]),
            "answerType": "integer",
        },
        {
            "prompt": "What is the database name?",
            "groundTruth": config["database"]["name"],
            "answerType": "string",
        },
        {
            "prompt": "What is the session refresh threshold?",
            "groundTruth": str(config["authentication"]["session"]["refreshThreshold"]),
            "answerType": "integer",
        },
        {
            "prompt": "What is the version in the configuration?",
            "groundTruth": config["version"],
            "answerType": "string",
        },
    ]
    for q in field_retrieval_questions[
        : QUESTION_LIMITS["nestedConfig"]["fieldRetrieval"]
    ]:
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(q["prompt"])
            .ground_truth(q["groundTruth"])
            .type("field-retrieval")
            .dataset("nested-config")
            .answer_type(cast(AnswerType, q["answerType"]))
            .build()
        )

    role_count = len(config["permissions"]["roles"])
    group_count = len(config["permissions"]["groups"])
    provider_count = len(config["authentication"]["providers"])
    feature_count = len(config["features"])
    replica_count = len(config["database"]["replicas"])

    questions.extend(
        [
            QuestionBuilder()
            .id(get_id())
            .prompt("How many roles are defined in permissions?")
            .ground_truth(str(role_count))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many groups are defined in permissions?")
            .ground_truth(str(group_count))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many authentication providers are configured?")
            .ground_truth(str(provider_count))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many feature flags are defined?")
            .ground_truth(str(feature_count))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many database replicas are configured?")
            .ground_truth(str(replica_count))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
        ]
    )

    admin_scope_provider_count = sum(
        1 for p in config["authentication"]["providers"] if "admin" in p["scopes"]
    )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt('How many authentication providers include the "admin" scope?')
        .ground_truth(str(admin_scope_provider_count))
        .type("aggregation")
        .dataset("nested-config")
        .answer_type("integer")
        .build()
    )

    enabled_features = sum(1 for f in config["features"].values() if f["enabled"])
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many feature flags are enabled?")
        .ground_truth(str(enabled_features))
        .type("aggregation")
        .dataset("nested-config")
        .answer_type("integer")
        .build()
    )

    admin_permissions = len(
        config["permissions"]["roles"].get("admin", {}).get("permissions", [])
    )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many permissions does the admin role have?")
        .ground_truth(str(admin_permissions))
        .type("aggregation")
        .dataset("nested-config")
        .answer_type("integer")
        .build()
    )

    total_permissions = sum(
        len(r["permissions"]) for r in config["permissions"]["roles"].values()
    )
    distinct_permissions = len(
        set(
            p for r in config["permissions"]["roles"].values() for p in r["permissions"]
        )
    )
    total_variants = sum(len(f["variants"]) for f in config["features"].values())
    high_priority_replicas = sum(
        1 for r in config["database"]["replicas"] if r["priority"] > 2
    )
    features_with_high_rollout = sum(
        1 for f in config["features"].values() if f["rollout"] > 50
    )
    groups_with_multiple_roles = sum(
        1 for g in config["permissions"]["groups"].values() if len(g["roles"]) > 1
    )

    questions.extend(
        [
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the total number of permissions across all roles?")
            .ground_truth(str(total_permissions))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many distinct permissions are defined across all roles?")
            .ground_truth(str(distinct_permissions))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the total number of variants across all feature flags?")
            .ground_truth(str(total_variants))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many database replicas have a priority greater than 2?")
            .ground_truth(str(high_priority_replicas))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many feature flags have a rollout percentage greater than 50?")
            .ground_truth(str(features_with_high_rollout))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many groups have more than one role assigned?")
            .ground_truth(str(groups_with_multiple_roles))
            .type("aggregation")
            .dataset("nested-config")
            .answer_type("integer")
            .build(),
        ]
    )

    filtering_questions = [
        {
            "prompt": "How many feature flags are enabled with rollout greater than 50%?",
            "groundTruth": str(
                sum(
                    1
                    for f in config["features"].values()
                    if f["enabled"] and f["rollout"] > 50
                )
            ),
        },
        {
            "prompt": "How many groups have the admin role?",
            "groundTruth": str(
                sum(
                    1
                    for g in config["permissions"]["groups"].values()
                    if "admin" in g["roles"]
                )
            ),
        },
        {
            "prompt": "How many database replicas have priority greater than 2 and port 5432?",
            "groundTruth": str(
                sum(
                    1
                    for r in config["database"]["replicas"]
                    if r["priority"] > 2 and r["port"] == 5432
                )
            ),
        },
        {
            "prompt": "How many authentication providers have more than 2 scopes?",
            "groundTruth": str(
                sum(
                    1
                    for p in config["authentication"]["providers"]
                    if len(p["scopes"]) > 2
                )
            ),
        },
        {
            "prompt": "How many roles have at least 5 permissions?",
            "groundTruth": str(
                sum(
                    1
                    for r in config["permissions"]["roles"].values()
                    if len(r["permissions"]) >= 5
                )
            ),
        },
        {
            "prompt": "How many feature flags are disabled with rollout less than 25%?",
            "groundTruth": str(
                sum(
                    1
                    for f in config["features"].values()
                    if not f["enabled"] and f["rollout"] < 25
                )
            ),
        },
        {
            "prompt": "How many enabled features have at least 2 variants?",
            "groundTruth": str(
                sum(
                    1
                    for f in config["features"].values()
                    if f["enabled"] and len(f["variants"]) >= 2
                )
            ),
        },
    ]
    for q in filtering_questions[: QUESTION_LIMITS["nestedConfig"]["filteringComplex"]]:
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(q["prompt"])
            .ground_truth(q["groundTruth"])
            .type("filtering")
            .dataset("nested-config")
            .answer_type("integer")
            .build()
        )

    return questions


def generate_structural_validation_questions(get_id: Callable) -> List[Question]:
    questions: List[Question] = []
    validation_fixtures = [
        {"dataset": "structural-validation-control", "isValid": True},
        {"dataset": "structural-validation-truncated", "isValid": False},
        {"dataset": "structural-validation-extra-rows", "isValid": False},
        {"dataset": "structural-validation-width-mismatch", "isValid": False},
        {"dataset": "structural-validation-missing-fields", "isValid": False},
    ]
    for fixture in validation_fixtures:
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt("Is this data complete and valid? Answer only YES or NO.")
            .ground_truth("YES" if fixture["isValid"] else "NO")
            .type("structural-validation")
            .dataset(cast(str, fixture["dataset"]))
            .answer_type("boolean")
            .build()
        )
    return questions


def generate_structure_questions(
    employees: List[Employee],
    orders: List[Order],
    metrics: List[AnalyticsMetric],
    repos: List[Repository],
    logs: List[EventLog],
    get_id: Callable,
) -> List[Question]:
    questions: List[Question] = []

    # Tabular dataset (Employees)
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many employees are in the dataset?")
        .ground_truth(str(len(employees)))
        .type("structure-awareness")
        .dataset("tabular")
        .answer_type("integer")
        .build()
    )
    employee_fields = "id,name,email,department,salary,yearsExperience,active"
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("List the field names for employees (comma-separated, in order).")
        .ground_truth(employee_fields)
        .type("structure-awareness")
        .dataset("tabular")
        .answer_type("csv-list-ordered")
        .build()
    )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("What is the 3rd field name for employees?")
        .ground_truth("email")
        .type("structure-awareness")
        .dataset("tabular")
        .answer_type("string")
        .build()
    )
    if employees:
        last_employee = employees[-1]
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the department of the last employee in the dataset?")
            .ground_truth(last_employee["department"])
            .type("structure-awareness")
            .dataset("tabular")
            .answer_type("string")
            .build()
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the name of the last employee in the dataset?")
            .ground_truth(last_employee["name"])
            .type("structure-awareness")
            .dataset("tabular")
            .answer_type("string")
            .build()
        )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many fields does each employee record have?")
        .ground_truth("7")
        .type("structure-awareness")
        .dataset("tabular")
        .answer_type("integer")
        .build()
    )

    # Nested dataset (Orders)
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many orders are in the dataset?")
        .ground_truth(str(len(orders)))
        .type("structure-awareness")
        .dataset("nested")
        .answer_type("integer")
        .build()
    )
    order_fields = "orderId,customer,items,subtotal,tax,total,status,orderDate"
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt(
            "List the top-level field names for orders (comma-separated, in order)."
        )
        .ground_truth(order_fields)
        .type("structure-awareness")
        .dataset("nested")
        .answer_type("csv-list-ordered")
        .build()
    )
    if orders:
        order_with_many_items = max(orders, key=lambda o: len(o["items"]))
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many items are in order {order_with_many_items['orderId']}?")
            .ground_truth(str(len(order_with_many_items["items"])))
            .type("structure-awareness")
            .dataset("nested")
            .answer_type("integer")
            .build()
        )
    item_fields = "sku,name,quantity,price"
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt(
            "What are the field names for items within orders (comma-separated, in order)?"
        )
        .ground_truth(item_fields)
        .type("structure-awareness")
        .dataset("nested")
        .answer_type("csv-list-ordered")
        .build()
    )
    if orders:
        last_order = orders[-1]
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the status of the last order in the dataset?")
            .ground_truth(last_order["status"])
            .type("structure-awareness")
            .dataset("nested")
            .answer_type("string")
            .build()
        )
    customer_fields = "id,name,email,phone"
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt(
            "What are the field names for customer objects within orders (comma-separated, in order)?"
        )
        .ground_truth(customer_fields)
        .type("structure-awareness")
        .dataset("nested")
        .answer_type("csv-list-ordered")
        .build()
    )

    # Analytics dataset (Metrics)
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many metric records are in the dataset?")
        .ground_truth(str(len(metrics)))
        .type("structure-awareness")
        .dataset("analytics")
        .answer_type("integer")
        .build()
    )
    metric_fields = "date,views,clicks,conversions,revenue,bounceRate"
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("List the field names for metrics (comma-separated, in order).")
        .ground_truth(metric_fields)
        .type("structure-awareness")
        .dataset("analytics")
        .answer_type("csv-list-ordered")
        .build()
    )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("What is the 5th field name for analytics metrics?")
        .ground_truth("revenue")
        .type("structure-awareness")
        .dataset("analytics")
        .answer_type("string")
        .build()
    )
    if metrics:
        last_metric = metrics[-1]
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the date of the last metric record in the dataset?")
            .ground_truth(last_metric["date"])
            .type("structure-awareness")
            .dataset("analytics")
            .answer_type("string")
            .build()
        )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many fields does each metric record have?")
        .ground_truth("6")
        .type("structure-awareness")
        .dataset("analytics")
        .answer_type("integer")
        .build()
    )

    # GitHub dataset (Repositories)
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many repositories are in the dataset?")
        .ground_truth(str(len(repos)))
        .type("structure-awareness")
        .dataset("github")
        .answer_type("integer")
        .build()
    )
    repo_fields = "id,name,repo,description,stars,watchers,forks,defaultBranch,createdAt,updatedAt,pushedAt"
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("List the field names for repositories (comma-separated, in order).")
        .ground_truth(repo_fields)
        .type("structure-awareness")
        .dataset("github")
        .answer_type("csv-list-ordered")
        .build()
    )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("What is the 7th field name for GitHub repositories?")
        .ground_truth("forks")
        .type("structure-awareness")
        .dataset("github")
        .answer_type("string")
        .build()
    )
    if repos:
        last_repo = repos[-1]
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the name of the last repository in the dataset?")
            .ground_truth(last_repo["name"])
            .type("structure-awareness")
            .dataset("github")
            .answer_type("string")
            .build()
        )
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many fields does each repository record have?")
        .ground_truth("11")
        .type("structure-awareness")
        .dataset("github")
        .answer_type("integer")
        .build()
    )

    # Event Logs dataset
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt("How many log entries are in the dataset?")
        .ground_truth(str(len(logs)))
        .type("structure-awareness")
        .dataset("event-logs")
        .answer_type("integer")
        .build()
    )
    log_fields = "timestamp,level,endpoint,statusCode,responseTime,userId,error"
    questions.append(
        QuestionBuilder()
        .id(get_id())
        .prompt(
            "List the field names for log entries (comma-separated, any order, including optional fields)."
        )
        .ground_truth(log_fields)
        .type("structure-awareness")
        .dataset("event-logs")
        .answer_type("csv-list-unordered")
        .build()
    )
    if logs:
        last_log = logs[-1]
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the level of the last log entry in the dataset?")
            .ground_truth(last_log["level"])
            .type("structure-awareness")
            .dataset("event-logs")
            .answer_type("string")
            .build()
        )

    return questions


def generate_tabular_questions(
    employees: List[Employee], get_id: Callable
) -> List[Question]:
    questions: List[Question] = []
    field_generators = [
        lambda emp, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the salary of {emp['name']}?")
        .ground_truth(str(emp["salary"]))
        .type("field-retrieval")
        .dataset("tabular")
        .answer_type("integer")
        .build(),
        lambda emp, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What department does {emp['name']} work in?")
        .ground_truth(emp["department"])
        .type("field-retrieval")
        .dataset("tabular")
        .answer_type("string")
        .build(),
        lambda emp, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"What is the email address of {emp['name']}?")
        .ground_truth(emp["email"])
        .type("field-retrieval")
        .dataset("tabular")
        .answer_type("string")
        .build(),
        lambda emp, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"How many years of experience does {emp['name']} have?")
        .ground_truth(str(emp["yearsExperience"]))
        .type("field-retrieval")
        .dataset("tabular")
        .answer_type("integer")
        .build(),
        lambda emp, get_id: QuestionBuilder()
        .id(get_id())
        .prompt(f"Is {emp['name']} an active employee?")
        .ground_truth("yes" if emp["active"] else "no")
        .type("field-retrieval")
        .dataset("tabular")
        .answer_type("boolean")
        .build(),
    ]
    questions.extend(
        rotate_questions(
            employees,
            field_generators,
            QUESTION_LIMITS["tabular"]["fieldRetrieval"],
            SAMPLE_STRIDES["EMPLOYEE_FIELD"],
            get_id,
        )
    )

    departments = sorted(list(set(e["department"] for e in employees)))
    for dept in departments[: QUESTION_LIMITS["tabular"]["aggregationDepartments"]]:
        count = sum(1 for e in employees if e["department"] == dept)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many employees work in {dept}?")
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("tabular")
            .answer_type("integer")
            .build()
        )

    for threshold in QUESTION_THRESHOLDS["tabular"]["salaryRanges"]:
        count = sum(1 for e in employees if e["salary"] > threshold)
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many employees have a salary greater than {threshold}?")
            .ground_truth(str(count))
            .type("aggregation")
            .dataset("tabular")
            .answer_type("integer")
            .build()
        )

    total_employees = len(employees)
    avg_salary = (
        round(sum(e["salary"] for e in employees) / total_employees)
        if total_employees > 0
        else 0
    )
    active_count = sum(1 for e in employees if e["active"])
    inactive_count = total_employees - active_count

    questions.extend(
        [
            QuestionBuilder()
            .id(get_id())
            .prompt("How many employees are in the dataset?")
            .ground_truth(str(total_employees))
            .type("aggregation")
            .dataset("tabular")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("What is the average salary across all employees?")
            .ground_truth(str(avg_salary))
            .type("aggregation")
            .dataset("tabular")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many employees are active?")
            .ground_truth(str(active_count))
            .type("aggregation")
            .dataset("tabular")
            .answer_type("integer")
            .build(),
            QuestionBuilder()
            .id(get_id())
            .prompt("How many employees are inactive?")
            .ground_truth(str(inactive_count))
            .type("aggregation")
            .dataset("tabular")
            .answer_type("integer")
            .build(),
        ]
    )

    for dept in departments[
        : QUESTION_LIMITS["tabular"]["filteringMultiConditionDepartments"]
    ]:
        count = sum(
            1
            for e in employees
            if e["department"] == dept
            and e["salary"]
            > QUESTION_THRESHOLDS["tabular"]["departmentSalaryThreshold"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many employees in {dept} have a salary greater than {QUESTION_THRESHOLDS['tabular']['departmentSalaryThreshold']}?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("tabular")
            .answer_type("integer")
            .build()
        )

    for exp in QUESTION_THRESHOLDS["tabular"]["experienceYears"][
        : QUESTION_LIMITS["tabular"]["filteringExperience"]
    ]:
        count = sum(1 for e in employees if e["yearsExperience"] > exp and e["active"])
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many active employees have more than {exp} years of experience?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("tabular")
            .answer_type("integer")
            .build()
        )

    for dept in departments[: QUESTION_LIMITS["tabular"]["filteringDepartmentExp"]]:
        count = sum(
            1
            for e in employees
            if e["department"] == dept
            and e["yearsExperience"]
            > QUESTION_THRESHOLDS["tabular"]["departmentExperienceThreshold"]
        )
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(
                f"How many employees in {dept} have more than {QUESTION_THRESHOLDS['tabular']['departmentExperienceThreshold']} years of experience?"
            )
            .ground_truth(str(count))
            .type("filtering")
            .dataset("tabular")
            .answer_type("integer")
            .build()
        )

    for dept in departments[: QUESTION_LIMITS["tabular"]["filteringDepartmentActive"]]:
        count = sum(1 for e in employees if e["department"] == dept and e["active"])
        questions.append(
            QuestionBuilder()
            .id(get_id())
            .prompt(f"How many active employees work in {dept}?")
            .ground_truth(str(count))
            .type("filtering")
            .dataset("tabular")
            .answer_type("integer")
            .build()
        )

    return questions


def generate_questions(toon_datasets: List[Dataset]) -> List[Question]:
    questions: List[Question] = []
    id_gen = create_id_generator()

    def get_id() -> str:
        return next(id_gen)

    datasets = {d["name"]: d["data"] for d in toon_datasets}

    tabular_data = datasets.get("tabular", {}).get("employees", [])
    nested_data = datasets.get("nested", {}).get("orders", [])
    analytics_data = datasets.get("analytics", {}).get("metrics", [])
    github_data = datasets.get("github", {}).get("repositories", [])
    event_logs_data = datasets.get("event-logs", {}).get("logs", [])
    nested_config_data = datasets.get("nested-config")

    questions.extend(generate_tabular_questions(tabular_data, get_id))
    questions.extend(generate_nested_questions(nested_data, get_id))
    questions.extend(generate_analytics_questions(analytics_data, get_id))
    questions.extend(generate_github_questions(github_data, get_id))
    questions.extend(generate_event_logs_questions(event_logs_data, get_id))
    if nested_config_data:
        questions.extend(
            generate_nested_config_questions(
                cast(NestedConfig, nested_config_data), get_id
            )
        )
    questions.extend(
        generate_structure_questions(
            tabular_data,
            nested_data,
            analytics_data,
            github_data,
            event_logs_data,
            get_id,
        )
    )
    questions.extend(generate_structural_validation_questions(get_id))

    return questions

from typing import List, Dict, Any, TypedDict, Literal, Generator, Callable, Optional, cast
from itertools import count
from src.benchmark_datasets import (
    Employee,
    Order,
    AnalyticsMetric,
    Repository,
    EventLog,
    NestedConfig,
    ACCURACY_DATASETS,
)

QuestionType = Literal[
    'field-retrieval',
    'retrieval',
    'aggregation',
    'filtering',
    'structure-awareness',
    'structural-validation',
]

AnswerType = Literal[
    'string',
    'integer',
    'number',
    'boolean',
    'csv-list-ordered',
    'csv-list-unordered',
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


QUESTION_THRESHOLDS = {
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
        if not all(k in self._question for k in ["id", "prompt", "groundTruth", "type", "dataset"]):
            raise ValueError("Incomplete question")
        return cast(Question, self._question)

def rotate_questions(items: List[Any], generators: List[Callable], limit: int, stride: int, get_id: Callable) -> List[Question]:
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


def generate_analytics_questions(metrics: List[AnalyticsMetric], get_id: Callable) -> List[Question]:
    questions: List[Question] = []

    metric_field_generators = [
        lambda metric, get_id: QuestionBuilder().id(get_id()).prompt(f"What are the views for {metric['date']}?").ground_truth(str(metric['views'])).type('field-retrieval').dataset('analytics').answer_type('integer').build(),
        lambda metric, get_id: QuestionBuilder().id(get_id()).prompt(f"What is the revenue for {metric['date']}?").ground_truth(str(metric['revenue'])).type('field-retrieval').dataset('analytics').answer_type('number').normalize({"decimal_places": 2}).build(),
        lambda metric, get_id: QuestionBuilder().id(get_id()).prompt(f"What is the bounce rate for {metric['date']}?").ground_truth(str(metric['bounceRate'])).type('field-retrieval').dataset('analytics').answer_type('number').normalize({"decimal_places": 2}).build(),
        lambda metric, get_id: QuestionBuilder().id(get_id()).prompt(f"How many conversions were there on {metric['date']}?").ground_truth(str(metric['conversions'])).type('field-retrieval').dataset('analytics').answer_type('integer').build(),
    ]
    questions.extend(rotate_questions(metrics, metric_field_generators, QUESTION_LIMITS['analytics']['fieldRetrievalDates'], SAMPLE_STRIDES['ANALYTICS_FIELD'], get_id))

    total_days = len(metrics)
    total_views = sum(m['views'] for m in metrics)
    total_conversions = sum(m['conversions'] for m in metrics)
    total_revenue = sum(m['revenue'] for m in metrics)
    avg_bounce_rate = sum(m['bounceRate'] for m in metrics) / total_days if total_days > 0 else 0

    questions.extend([
        QuestionBuilder().id(get_id()).prompt('How many days of data are in the dataset?').ground_truth(str(total_days)).type('aggregation').dataset('analytics').answer_type('integer').build(),
        QuestionBuilder().id(get_id()).prompt('What is the total number of views across all dates?').ground_truth(str(total_views)).type('aggregation').dataset('analytics').answer_type('integer').build(),
        QuestionBuilder().id(get_id()).prompt('What is the total number of conversions across all dates?').ground_truth(str(total_conversions)).type('aggregation').dataset('analytics').answer_type('integer').build(),
        QuestionBuilder().id(get_id()).prompt('What is the total revenue across all dates?').ground_truth(f"{total_revenue:.2f}").type('aggregation').dataset('analytics').answer_type('number').normalize({"decimal_places": 2}).build(),
        QuestionBuilder().id(get_id()).prompt('What is the average bounce rate?').ground_truth(f"{avg_bounce_rate:.2f}").type('aggregation').dataset('analytics').answer_type('number').normalize({"decimal_places": 2}).build(),
    ])

    for threshold in cast(Dict[str, Any], QUESTION_THRESHOLDS)['analytics']['views']:
        count = sum(1 for m in metrics if m['views'] > threshold)
        questions.append(QuestionBuilder().id(get_id()).prompt(f"How many days had more than {threshold} views?").ground_truth(str(count)).type('aggregation').dataset('analytics').answer_type('integer').build())

    return questions

def generate_event_logs_questions(logs: List[EventLog], get_id: Callable) -> List[Question]:
    questions: List[Question] = []
    # This is not a full implementation, just a sample.
    total_logs = len(logs)
    questions.append(QuestionBuilder().id(get_id()).prompt("How many log entries are in the dataset?").ground_truth(str(total_logs)).type('aggregation').dataset('event-logs').answer_type('integer').build())
    return questions

def generate_github_questions(repos: List[Repository], get_id: Callable) -> List[Question]:
    questions: List[Question] = []
    # This is not a full implementation, just a sample.
    total_repos = len(repos)
    questions.append(QuestionBuilder().id(get_id()).prompt("How many repositories are in the dataset?").ground_truth(str(total_repos)).type('aggregation').dataset('github').answer_type('integer').build())
    return questions

def generate_nested_questions(orders: List[Order], get_id: Callable) -> List[Question]:
    questions: List[Question] = []
    # This is not a full implementation, just a sample.
    total_orders = len(orders)
    questions.append(QuestionBuilder().id(get_id()).prompt("How many orders are in the dataset?").ground_truth(str(total_orders)).type('aggregation').dataset('nested').answer_type('integer').build())
    return questions

def generate_nested_config_questions(config: NestedConfig, get_id: Callable) -> List[Question]:
    questions: List[Question] = []
    # This is not a full implementation, just a sample.
    questions.append(QuestionBuilder().id(get_id()).prompt("What is the environment in the configuration?").ground_truth(config['environment']).type('field-retrieval').dataset('nested-config').answer_type('string').build())
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
    # This is not a full implementation, just a sample.
    questions.append(QuestionBuilder().id(get_id()).prompt("How many employees are in the dataset?").ground_truth(str(len(employees))).type('structure-awareness').dataset('tabular').answer_type('integer').build())
    return questions

def generate_tabular_questions(employees: List[Employee], get_id: Callable) -> List[Question]:
    questions: List[Question] = []
    # This is not a full implementation, just a sample.
    field_generators = [
        lambda emp, get_id: QuestionBuilder().id(get_id()).prompt(f"What is the salary of {emp['name']}?").ground_truth(str(emp['salary'])).type('field-retrieval').dataset('tabular').answer_type('integer').build(),
    ]
    questions.extend(rotate_questions(employees, field_generators, QUESTION_LIMITS['tabular']['fieldRetrieval'], SAMPLE_STRIDES['EMPLOYEE_FIELD'], get_id))
    return questions


def generate_questions() -> List[Question]:
    questions: List[Question] = []
    id_gen = create_id_generator()
    def get_id() -> str:
        return next(id_gen)

    datasets = {d["name"]: d["data"] for d in ACCURACY_DATASETS}

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
        questions.extend(generate_nested_config_questions(cast(NestedConfig, nested_config_data), get_id))
    questions.extend(generate_structure_questions(tabular_data, nested_data, analytics_data, github_data, event_logs_data, get_id))
    questions.extend(generate_structural_validation_questions(get_id))

    return questions

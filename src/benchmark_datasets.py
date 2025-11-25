import json
from faker import Faker
from typing import List, Dict, Any, TypedDict, Literal, cast

# Seed for reproducibility
fake = Faker()
Faker.seed(12345)

# Load GitHub repos data
with open("data/github-repos.json") as f:
    github_repos = json.load(f)


class Employee(TypedDict):
    id: int
    name: str
    email: str
    department: str
    salary: int
    yearsExperience: int
    active: bool


class Order(TypedDict):
    orderId: str
    customer: Dict[str, Any]
    items: List[Dict[str, Any]]
    subtotal: float
    tax: float
    total: float
    status: str
    orderDate: str


class AnalyticsMetric(TypedDict):
    date: str
    views: int
    clicks: int
    conversions: int
    revenue: float
    bounceRate: float


class Repository(TypedDict):
    id: int
    name: str
    repo: str
    description: str
    stars: int
    watchers: int
    forks: int
    defaultBranch: str
    createdAt: str
    updatedAt: str
    pushedAt: str


class EventLog(TypedDict, total=False):
    timestamp: str
    level: Literal["info", "warn", "error"]
    endpoint: str
    statusCode: int
    responseTime: int
    userId: int
    error: Dict[str, Any]


class NestedConfig(TypedDict):
    environment: str
    version: str
    database: Dict[str, Any]
    features: Dict[str, Any]
    authentication: Dict[str, Any]
    permissions: Dict[str, Any]


class Product(TypedDict):
    sku: str
    name: str
    category: str
    price: float
    qty: int
    lastUpdated: str


class Dataset(TypedDict):
    name: str
    description: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]


def generate_employees(count: int) -> Dict[str, List[Employee]]:
    departments = ["Engineering", "Sales", "Marketing", "HR", "Operations", "Finance"]
    employees: List[Employee] = []
    for i in range(count):
        years_exp = fake.random_int(min=1, max=25)
        employee: Employee = {
            "id": i + 1,
            "name": fake.name(),
            "email": fake.email().lower(),
            "department": departments[i % len(departments)],
            "salary": fake.random_int(min=45000, max=150000),
            "yearsExperience": years_exp,
            "active": fake.boolean(chance_of_getting_true=80),
        }
        employees.append(employee)
    return {"employees": employees}


PRODUCT_NAMES = [
    "Wireless Mouse",
    "USB Cable",
    "Laptop Stand",
    "Keyboard",
    "Webcam",
    "Headphones",
    "Monitor",
    "Desk Lamp",
]
ORDER_STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]


def generate_orders(count: int) -> Dict[str, List[Order]]:
    orders: List[Order] = []
    for i in range(count):
        customer_id = (i % 20) + 1
        item_count = fake.random_int(min=1, max=4)

        items = []
        for j in range(item_count):
            price = round(
                fake.pyfloat(min_value=9.99, max_value=199.99, right_digits=2), 2
            )
            quantity = fake.random_int(min=1, max=5)
            item = {
                "sku": f"SKU-{fake.pystr(min_chars=6, max_chars=6).upper()}",
                "name": PRODUCT_NAMES[j % len(PRODUCT_NAMES)],
                "quantity": quantity,
                "price": price,
            }
            items.append(item)

        subtotal = round(sum(item["price"] * item["quantity"] for item in items), 2)
        tax = round(subtotal * 0.08, 2)
        total = round(subtotal + tax, 2)

        order: Order = {
            "orderId": f"ORD-{(i + 1):04d}",
            "customer": {
                "id": customer_id,
                "name": fake.name(),
                "email": fake.email().lower(),
                "phone": fake.phone_number(),
            },
            "items": items,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "status": ORDER_STATUSES[i % len(ORDER_STATUSES)],
            "orderDate": fake.date_this_decade().isoformat(),
        }
        orders.append(order)
    return {"orders": orders}


def generate_analytics_data(
    days: int, start_date_str: str = "2025-01-01"
) -> Dict[str, List[AnalyticsMetric]]:
    from datetime import datetime, timedelta

    start_date = datetime.fromisoformat(start_date_str)
    metrics = []
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        base_views = 5000
        weekend_multiplier = 0.7 if current_date.weekday() >= 5 else 1.0
        views = round(
            base_views * weekend_multiplier + fake.random_int(min=-1000, max=3000)
        )
        clicks = round(views * fake.pyfloat(min_value=0.02, max_value=0.08))
        conversions = round(clicks * fake.pyfloat(min_value=0.05, max_value=0.15))
        avg_order_value = fake.pyfloat(min_value=49.99, max_value=299.99)
        revenue = round(conversions * avg_order_value, 2)

        metric: AnalyticsMetric = {
            "date": current_date.isoformat().split("T")[0],
            "views": views,
            "clicks": clicks,
            "conversions": conversions,
            "revenue": revenue,
            "bounceRate": round(fake.pyfloat(min_value=0.3, max_value=0.7), 2),
        }
        metrics.append(metric)
    return {"metrics": metrics}


def generate_event_logs(count: int) -> Dict[str, List[EventLog]]:
    endpoints = [
        "/api/users",
        "/api/orders",
        "/api/products",
        "/api/auth",
        "/api/payments",
    ]
    levels = ["info", "warn", "error"]
    logs = []
    for _ in range(count):
        level = cast(
            Literal["info", "warn", "error"], fake.random_element(elements=levels)
        )
        has_error = level == "error" or (
            level == "warn" and fake.boolean(chance_of_getting_true=30)
        )

        log: EventLog = {
            "timestamp": fake.iso8601(),
            "level": level,
            "endpoint": fake.random_element(elements=endpoints),
            "statusCode": fake.random_int(min=400, max=599)
            if has_error
            else fake.random_int(min=200, max=299),
            "responseTime": fake.random_int(min=10, max=5000),
            "userId": fake.random_int(min=1000, max=9999),
        }

        if has_error:
            log["error"] = {
                "message": fake.random_element(
                    elements=[
                        "Database connection timeout",
                        "Invalid authentication token",
                        "Resource not found",
                        "Internal server error",
                        "Rate limit exceeded",
                    ]
                ),
                "stack": f"Error: {fake.sentence()}\n  at {fake.word()}\n  at {fake.word()}",
                "retryable": fake.boolean(chance_of_getting_true=60),
            }
        logs.append(log)
    return {"logs": logs}


def generate_nested_config() -> NestedConfig:
    return {
        "environment": fake.random_element(
            elements=["production", "staging", "development"]
        ),
        "version": "1.2.3",
        "database": {
            "host": fake.domain_name(),
            "port": 5432,
            "name": fake.word(),
            "pool": {
                "min": 2,
                "max": fake.random_int(min=10, max=50),
                "idleTimeout": 30000,
            },
            "replicas": [
                {
                    "host": f"replica-{i + 1}.{fake.domain_name()}",
                    "port": 5432,
                    "priority": i + 1,
                }
                for i in range(3)
            ],
        },
        "features": {
            "darkMode": {
                "enabled": fake.boolean(),
                "rollout": fake.random_int(min=0, max=100),
                "variants": [
                    {
                        "name": "default",
                        "weight": 70,
                        "config": {"theme": "dark", "animations": True},
                    },
                    {
                        "name": "minimal",
                        "weight": 30,
                        "config": {"theme": "dark", "animations": False},
                    },
                ],
            },
            "analytics": {
                "enabled": fake.boolean(),
                "rollout": fake.random_int(min=0, max=100),
                "variants": [
                    {
                        "name": "full",
                        "weight": 100,
                        "config": {"tracking": "all", "sampling": 1.0},
                    },
                ],
            },
        },
        "authentication": {
            "providers": [
                {
                    "name": "oauth2",
                    "clientId": fake.uuid4(),
                    "scopes": ["read", "write", "admin"],
                    "config": {
                        "authUrl": fake.uri(),
                        "tokenUrl": fake.uri(),
                    },
                },
                {
                    "name": "saml",
                    "clientId": fake.uuid4(),
                    "scopes": ["read"],
                    "config": {
                        "entryPoint": fake.uri(),
                        "cert": fake.pystr(min_chars=64, max_chars=64),
                    },
                },
            ],
            "session": {
                "secret": fake.pystr(min_chars=32, max_chars=32),
                "duration": 86400,
                "refreshThreshold": 3600,
            },
        },
        "permissions": {
            "roles": {
                "admin": {
                    "permissions": [
                        "read",
                        "write",
                        "delete",
                        "manage_users",
                        "manage_roles",
                    ],
                    "inherits": [],
                },
                "editor": {
                    "permissions": ["read", "write"],
                    "inherits": ["viewer"],
                },
                "viewer": {
                    "permissions": ["read"],
                    "inherits": [],
                },
            },
            "groups": {
                "engineering": {
                    "members": [fake.email() for _ in range(5)],
                    "roles": ["admin", "editor"],
                },
                "support": {
                    "members": [fake.email() for _ in range(3)],
                    "roles": ["viewer"],
                },
            },
        },
    }


class StructuralValidationFixture(TypedDict):
    type: Literal["truncated", "extra-rows", "width-mismatch", "missing-fields"]
    description: str
    data: Dict[str, Any]
    isValid: bool


def generate_structural_validation_fixtures() -> List[StructuralValidationFixture]:
    base_data = generate_employees(20)

    fixtures: List[StructuralValidationFixture] = [
        {
            "type": "truncated",
            "description": "Valid complete dataset (control)",
            "data": {"employees": base_data["employees"]},
            "isValid": True,
        },
        {
            "type": "truncated",
            "description": "Array truncated: 3 rows removed from end",
            "data": {"employees": base_data["employees"][:-3]},
            "isValid": False,
        },
        {
            "type": "extra-rows",
            "description": "Extra rows added beyond declared length",
            "data": {
                "employees": base_data["employees"]
                + generate_employees(3)["employees"],
            },
            "isValid": False,
        },
        {
            "type": "width-mismatch",
            "description": "Inconsistent field count (missing salary in row 10)",
            "data": {
                "employees": [
                    {k: v for k, v in emp.items() if k != "salary"} if i == 9 else emp
                    for i, emp in enumerate(base_data["employees"])
                ],
            },
            "isValid": False,
        },
        {
            "type": "missing-fields",
            "description": "Missing required fields (no email in multiple rows)",
            "data": {
                "employees": [
                    {k: v for k, v in emp.items() if k != "email"}
                    if i % 5 == 0
                    else emp
                    for i, emp in enumerate(base_data["employees"])
                ],
            },
            "isValid": False,
        },
    ]
    return fixtures


dataset_names = [
    "structural-validation-control",
    "structural-validation-truncated",
    "structural-validation-extra-rows",
    "structural-validation-width-mismatch",
    "structural-validation-missing-fields",
]


def get_toon_datasets(fullsize_ratio: float = 1.0) -> List[Dataset]:
    """Generates the toon benchmark datasets, with sizes scaled by the given ratio."""

    tabular_dataset: Dataset = {
        "name": "tabular",
        "description": "Uniform employee records",
        "data": generate_employees(int(100 * fullsize_ratio)),
        "metadata": {
            "supportsCSV": True,
            "structureClass": "uniform",
            "tabularEligibility": 100,
        },
    }

    nested_dataset: Dataset = {
        "name": "nested",
        "description": "E-commerce orders with nested structures",
        "data": generate_orders(int(50 * fullsize_ratio)),
        "metadata": {
            "supportsCSV": False,
            "structureClass": "nested",
            "tabularEligibility": 33,
        },
    }

    analytics_dataset: Dataset = {
        "name": "analytics",
        "description": "Time-series analytics data",
        "data": generate_analytics_data(int(60 * fullsize_ratio)),
        "metadata": {
            "supportsCSV": True,
            "structureClass": "uniform",
            "tabularEligibility": 100,
        },
    }

    github_dataset: Dataset = {
        "name": "github",
        "description": "Top 100 GitHub repositories",
        "data": {
            "repositories": github_repos[: int(100 * fullsize_ratio)],
        },
        "metadata": {
            "supportsCSV": True,
            "structureClass": "uniform",
            "tabularEligibility": 100,
        },
    }

    event_logs_dataset: Dataset = {
        "name": "event-logs",
        "description": "Semi-uniform event logs",
        "data": generate_event_logs(int(75 * fullsize_ratio)),
        "metadata": {
            "supportsCSV": False,
            "structureClass": "semi-uniform",
            "tabularEligibility": 50,
        },
    }

    nested_config_dataset: Dataset = {
        "name": "nested-config",
        "description": "Deeply nested configuration",
        "data": generate_nested_config(),
        "metadata": {
            "supportsCSV": False,
            "structureClass": "deep",
            "tabularEligibility": 0,
        },
    }

    structural_validation_datasets: List[Dataset] = cast(
        List[Dataset],
        [
            {
                "name": dataset_names[i],
                "description": fixture["description"],
                "data": fixture["data"],
                "metadata": {
                    "supportsCSV": True,
                    "structureClass": "uniform",
                    "tabularEligibility": 100,
                },
            }
            for i, fixture in enumerate(generate_structural_validation_fixtures())
        ],
    )

    return [
        tabular_dataset,
        nested_dataset,
        analytics_dataset,
        github_dataset,
        event_logs_dataset,
        nested_config_dataset,
        *structural_validation_datasets,
    ]


TOON_DATASETS: List[Dataset] = get_toon_datasets()

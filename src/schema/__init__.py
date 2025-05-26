import importlib

schema_base = None
try:
    importlib.import_module("src.schema.models")
    schema_base = "src.schema"
except ImportError:
    schema_base = "schema"

models = importlib.import_module(f"{schema_base}.models")
schema = importlib.import_module(f"{schema_base}.schema")

AllModelEnum = models.AllModelEnum

AgentInfo = schema.AgentInfo
ChatHistory = schema.ChatHistory
ChatHistoryInput = schema.ChatHistoryInput
ChatMessage = schema.ChatMessage
Feedback = schema.Feedback
FeedbackResponse = schema.FeedbackResponse
ServiceMetadata = schema.ServiceMetadata
StreamInput = schema.StreamInput
UserInput = schema.UserInput
AnnotationsRequest = schema.AnnotationsRequest
AnnotationsResponse = schema.AnnotationsResponse
DebugBlocksRequest = schema.DebugBlocksRequest
AnnotationItem = schema.AnnotationItem
DocumentSourceInfo = schema.DocumentSourceInfo
DocumentSourceResponse = schema.DocumentSourceResponse

__all__ = [
    "AgentInfo",
    "AllModelEnum",
    "UserInput",
    "ChatMessage",
    "ServiceMetadata",
    "StreamInput",
    "Feedback",
    "FeedbackResponse",
    "ChatHistoryInput",
    "ChatHistory",
    "AnnotationsRequest",
    "AnnotationsResponse",
    "DebugBlocksRequest",
    "AnnotationItem",
    "DocumentSourceInfo",
    "DocumentSourceResponse",
]
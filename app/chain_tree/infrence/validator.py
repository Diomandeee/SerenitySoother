from typing import Dict, Any


class MessageIDValidator:
    @staticmethod
    def validate_message_id(message_id: str) -> bool:
        if not isinstance(message_id, str) or len(message_id) == 0:
            return False
        return True

    @staticmethod
    def validate_message_id_in_mapping(
        message_id: str, mapping: Dict[str, Any]
    ) -> bool:
        if not MessageIDValidator.validate_message_id(message_id):
            return False
        return message_id in mapping

    def validate_message_id_in_specific_mapping(
        message_id: str, specific_mapping: Dict[str, Any]
    ) -> bool:
        return MessageIDValidator.validate_message_id_in_mapping(
            message_id, specific_mapping
        )

    def validate_merge_point(self, message_id: str, mapping: Dict[str, Any]) -> bool:
        if not MessageIDValidator.validate_message_id_in_mapping(message_id, mapping):
            return False
        return len(mapping[message_id]["children"]) > 1

    def validate_merge_point_id(
        message_id: str, merge_point_id: str, mapping: Dict[str, Any]
    ) -> bool:
        if not MessageIDValidator.validate_message_id_in_mapping(message_id, mapping):
            return False
        if not MessageIDValidator.validate_message_id_in_mapping(
            merge_point_id, mapping
        ):
            return False
        if merge_point_id == message_id:
            return False
        return True

    def validate_merge_point_id_in_specific_mapping(
        message_id: str, merge_point_id: str, specific_mapping: Dict[str, Any]
    ) -> bool:
        return MessageIDValidator.validate_merge_point_id(
            message_id, merge_point_id, specific_mapping
        )

    def validate_merge_point_id_in_mapping(
        self, message_id: str, merge_point_id: str, mapping: Dict[str, Any]
    ) -> bool:
        if not MessageIDValidator.validate_merge_point_id(
            message_id, merge_point_id, mapping
        ):
            return False
        if not self.validate_merge_point(message_id, mapping):
            return False
        return True

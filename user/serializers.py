from rest_framework import serializers
from djoser.serializers import UserSerializer as DjoserUserSerializer
from djoser.serializers import UserCreateSerializer as DjoserUserCreateSerializer
from .models import User


class UserSerializer(DjoserUserSerializer):
    """
    Inherit from djoser's UserSerializer for user serialization.
    Add any custom fields or methods here if needed.
    """

    class Meta(DjoserUserSerializer.Meta):
        model = User
        fields = [
            "id",
            "email",
            "first_name",
            "last_name",
            "is_staff",
            "is_superuser",
            "is_active",
            "date_joined",
            "last_login",
        ]
        read_only_fields = ["id", "date_joined", "last_login"]


class UserCreateSerializer(DjoserUserCreateSerializer):
    """
    Inherit from djoser's UserCreateSerializer for user creation.
    Customize registration fields here if needed.
    """

    class Meta(DjoserUserCreateSerializer.Meta):
        model = User
        fields = [
            "email",
            "password",
            "first_name",
            "last_name",
        ]
        extra_kwargs = {"password": {"write_only": True}}

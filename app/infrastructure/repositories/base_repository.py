"""
Base Repository - Abstract repository pattern implementation.

Provides common CRUD operations and query helpers for all entities.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Type
from sqlalchemy.orm import Session

from app.models import Base

T = TypeVar('T', bound=Base)


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository providing common data access operations.

    Type Parameters:
        T: The SQLAlchemy model type this repository manages
    """

    def __init__(self, session: Session, model_class: Type[T]):
        """
        Initialize the repository.

        Args:
            session: SQLAlchemy database session
            model_class: The model class this repository manages
        """
        self.session = session
        self.model_class = model_class

    def get_by_id(self, entity_id: int) -> Optional[T]:
        """
        Retrieve an entity by its primary key.

        Args:
            entity_id: Primary key value

        Returns:
            The entity if found, None otherwise
        """
        return self.session.query(self.model_class).filter(
            self.model_class.id == entity_id
        ).first()

    def get_by_uuid(self, uuid: str) -> Optional[T]:
        """
        Retrieve an entity by its UUID.

        Args:
            uuid: UUID string

        Returns:
            The entity if found, None otherwise
        """
        if not hasattr(self.model_class, 'uuid'):
            raise AttributeError(f"{self.model_class.__name__} does not have a uuid field")
        return self.session.query(self.model_class).filter(
            self.model_class.uuid == uuid
        ).first()

    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """
        Retrieve all entities with optional pagination.

        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip

        Returns:
            List of entities
        """
        query = self.session.query(self.model_class).offset(offset)
        if limit:
            query = query.limit(limit)
        return query.all()

    def count(self) -> int:
        """
        Count total entities.

        Returns:
            Total count of entities
        """
        return self.session.query(self.model_class).count()

    def add(self, entity: T) -> T:
        """
        Add a new entity to the session.

        Args:
            entity: Entity to add

        Returns:
            The added entity
        """
        self.session.add(entity)
        return entity

    def add_all(self, entities: List[T]) -> List[T]:
        """
        Add multiple entities to the session.

        Args:
            entities: List of entities to add

        Returns:
            The added entities
        """
        self.session.add_all(entities)
        return entities

    def update(self, entity: T) -> T:
        """
        Update an entity (merge into session).

        Args:
            entity: Entity with updated values

        Returns:
            The merged entity
        """
        return self.session.merge(entity)

    def delete(self, entity: T) -> None:
        """
        Delete an entity.

        Args:
            entity: Entity to delete
        """
        self.session.delete(entity)

    def delete_by_id(self, entity_id: int) -> bool:
        """
        Delete an entity by its primary key.

        Args:
            entity_id: Primary key value

        Returns:
            True if entity was found and deleted, False otherwise
        """
        entity = self.get_by_id(entity_id)
        if entity:
            self.delete(entity)
            return True
        return False

    def commit(self) -> None:
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.session.rollback()

    def flush(self) -> None:
        """Flush pending changes to the database."""
        self.session.flush()

    def refresh(self, entity: T) -> T:
        """
        Refresh an entity from the database.

        Args:
            entity: Entity to refresh

        Returns:
            The refreshed entity
        """
        self.session.refresh(entity)
        return entity

    @abstractmethod
    def exists(self, **criteria) -> bool:
        """
        Check if an entity matching the criteria exists.

        Args:
            **criteria: Field-value pairs to match

        Returns:
            True if entity exists, False otherwise
        """
        pass

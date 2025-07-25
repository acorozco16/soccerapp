from sqlalchemy import Column, String, DateTime, Float, Integer, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class ProcessingStatus(enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class Video(Base):
    __tablename__ = "videos"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.UPLOADED)
    upload_time = Column(DateTime, default=datetime.utcnow)
    process_start_time = Column(DateTime, nullable=True)
    process_end_time = Column(DateTime, nullable=True)
    results_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Analysis results
    total_touches = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)
    video_duration = Column(Float, nullable=True)
    touches_per_minute = Column(Float, nullable=True)
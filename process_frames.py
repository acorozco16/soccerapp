import sys
  sys.path.append("training_data/processors")
  from frame_processor import FrameProcessor

  processor = FrameProcessor(
      scraped_data_dir="training_data/collected_data/youtube/scraper_0",
      output_dir="training_data/processed_dataset"
  )

  summary = processor.process_all_frames()
  print("Processing complete!")

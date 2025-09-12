from frame_processor import FrameProcessor

  processor = FrameProcessor(
      scraped_data_dir='../collected_data/youtube/scraper_0',
      output_dir='../processed_dataset'
  )

  summary = processor.process_all_frames()
  print('Processing complete!')

#!/usr/bin/env python3
"""
Test All Sample Videos
Runs analysis on all sample videos and compares with reference counts
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from video_processor import VideoProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reference data for sample videos
REFERENCE_DATA = {
    "clear_touches.mp4": {
        "manual_count": 23,
        "expected_range": (20, 26),
        "min_confidence": 0.80,
        "description": "Clear orange ball with good lighting"
    },
    "difficult_lighting.mp4": {
        "manual_count": 18,
        "expected_range": (13, 23),
        "min_confidence": 0.60,
        "description": "Poor lighting conditions"
    },
    "multiple_players.mp4": {
        "manual_count": 31,
        "expected_range": (26, 36),
        "min_confidence": 0.60,
        "description": "Multiple players, complex scene"
    },
    "fast_movement.mp4": {
        "manual_count": 15,
        "expected_range": (10, 20),
        "min_confidence": 0.50,
        "description": "Fast ball movement, motion blur"
    },
    "youth_soccer.mp4": {
        "manual_count": 27,
        "expected_range": (22, 32),
        "min_confidence": 0.80,
        "description": "Youth players, smaller ball"
    }
}


class SampleVideoTester:
    def __init__(self, sample_videos_dir: str = "../sample_videos"):
        self.sample_videos_dir = Path(sample_videos_dir)
        self.processor = VideoProcessor()
        self.results = []
    
    async def test_video(self, video_path: Path, reference_data: Dict) -> Dict:
        """Test a single video and compare with reference"""
        logger.info(f"Testing {video_path.name}...")
        
        try:
            # Generate video ID
            video_id = f"test_{video_path.stem}_{datetime.now().strftime('%H%M%S')}"
            
            # Run analysis
            results = await self.processor.analyze_video(str(video_path), video_id)
            
            # Extract metrics
            detected_count = results.get('total_ball_touches', 0)
            confidence_score = results.get('confidence_score', 0)
            processing_time = results.get('processing_time', 0)
            
            # Compare with reference
            manual_count = reference_data['manual_count']
            expected_range = reference_data['expected_range']
            min_confidence = reference_data['min_confidence']
            
            # Calculate accuracy metrics
            error = abs(detected_count - manual_count)
            error_percentage = (error / manual_count) * 100 if manual_count > 0 else 0
            within_range = expected_range[0] <= detected_count <= expected_range[1]
            confidence_ok = confidence_score >= min_confidence
            
            # Determine overall result
            passed = within_range and confidence_ok
            
            test_result = {
                'video_name': video_path.name,
                'description': reference_data['description'],
                'manual_count': manual_count,
                'detected_count': detected_count,
                'expected_range': expected_range,
                'error': error,
                'error_percentage': round(error_percentage, 1),
                'within_range': within_range,
                'confidence_score': confidence_score,
                'min_confidence': min_confidence,
                'confidence_ok': confidence_ok,
                'processing_time': processing_time,
                'passed': passed,
                'quality_assessment': results.get('quality_assessment', {}),
                'detection_summary': results.get('detection_summary', {})
            }
            
            logger.info(f"{video_path.name}: {detected_count}/{manual_count} touches, "
                       f"confidence {confidence_score:.3f}, {'PASS' if passed else 'FAIL'}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing {video_path.name}: {e}")
            return {
                'video_name': video_path.name,
                'error': str(e),
                'passed': False
            }
    
    async def test_all_videos(self) -> Dict:
        """Test all sample videos"""
        logger.info("Starting sample video testing...")
        
        # Find available videos
        available_videos = []
        for video_name in REFERENCE_DATA.keys():
            video_path = self.sample_videos_dir / video_name
            if video_path.exists():
                available_videos.append((video_path, REFERENCE_DATA[video_name]))
            else:
                logger.warning(f"Sample video not found: {video_path}")
        
        if not available_videos:
            logger.error("No sample videos found!")
            return {'error': 'No sample videos found'}
        
        logger.info(f"Found {len(available_videos)} sample videos to test")
        
        # Test each video
        test_results = []
        for video_path, reference_data in available_videos:
            result = await self.test_video(video_path, reference_data)
            test_results.append(result)
        
        # Calculate summary statistics
        passed_count = sum(1 for r in test_results if r.get('passed', False))
        total_count = len(test_results)
        
        # Calculate average metrics for passed tests
        passed_results = [r for r in test_results if r.get('passed', False)]
        if passed_results:
            avg_error = sum(r.get('error', 0) for r in passed_results) / len(passed_results)
            avg_error_pct = sum(r.get('error_percentage', 0) for r in passed_results) / len(passed_results)
            avg_confidence = sum(r.get('confidence_score', 0) for r in passed_results) / len(passed_results)
            avg_processing_time = sum(r.get('processing_time', 0) for r in passed_results) / len(passed_results)
        else:
            avg_error = avg_error_pct = avg_confidence = avg_processing_time = 0
        
        # Compile summary
        summary = {
            'test_run_info': {
                'timestamp': datetime.now().isoformat(),
                'total_videos': total_count,
                'passed': passed_count,
                'failed': total_count - passed_count,
                'pass_rate': (passed_count / total_count) * 100 if total_count > 0 else 0
            },
            'average_metrics': {
                'error': round(avg_error, 2),
                'error_percentage': round(avg_error_pct, 1),
                'confidence_score': round(avg_confidence, 3),
                'processing_time': round(avg_processing_time, 1)
            },
            'test_results': test_results
        }
        
        # Save results
        results_file = Path(f"sample_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {results_file}")
        return summary
    
    def print_summary(self, summary: Dict):
        """Print formatted test summary"""
        print("\n" + "="*60)
        print("SAMPLE VIDEO TEST RESULTS")
        print("="*60)
        
        info = summary['test_run_info']
        metrics = summary['average_metrics']
        
        print(f"\nOverall Results:")
        print(f"  Total Videos: {info['total_videos']}")
        print(f"  Passed: {info['passed']}")
        print(f"  Failed: {info['failed']}")
        print(f"  Pass Rate: {info['pass_rate']:.1f}%")
        
        print(f"\nAverage Performance (Passed Tests):")
        print(f"  Average Error: {metrics['error']:.1f} touches")
        print(f"  Average Error %: {metrics['error_percentage']:.1f}%")
        print(f"  Average Confidence: {metrics['confidence_score']:.3f}")
        print(f"  Average Processing Time: {metrics['processing_time']:.1f}s")
        
        print(f"\nDetailed Results:")
        print(f"{'Video':<25} {'Manual':<6} {'Detected':<8} {'Error':<5} {'Conf':<6} {'Status':<6}")
        print("-" * 60)
        
        for result in summary['test_results']:
            if 'error' in result and 'manual_count' not in result:
                print(f"{result['video_name']:<25} {'ERROR':<6} {'N/A':<8} {'N/A':<5} {'N/A':<6} {'FAIL':<6}")
            else:
                status = "PASS" if result['passed'] else "FAIL"
                print(f"{result['video_name']:<25} "
                      f"{result['manual_count']:<6} "
                      f"{result['detected_count']:<8} "
                      f"{result['error']:<5} "
                      f"{result['confidence_score']:<6.3f} "
                      f"{status:<6}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if info['pass_rate'] < 80:
            print("  ⚠️  Pass rate below 80% - consider tuning detection parameters")
        if metrics['error_percentage'] > 25:
            print("  ⚠️  High error rate - check ball detection accuracy")
        if metrics['confidence_score'] < 0.7:
            print("  ⚠️  Low confidence scores - improve detection reliability")
        
        if info['pass_rate'] >= 80 and metrics['error_percentage'] <= 20:
            print("  ✅ System performance is within acceptable ranges!")


async def main():
    """Main testing function"""
    tester = SampleVideoTester()
    
    try:
        summary = await tester.test_all_videos()
        tester.print_summary(summary)
        
        # Return exit code based on results
        pass_rate = summary['test_run_info']['pass_rate']
        return 0 if pass_rate >= 60 else 1  # 60% minimum pass rate
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
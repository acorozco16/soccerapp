import api from './api';
import { API_ENDPOINTS } from '../constants/config';

class DrillService {
  async getAvailableDrills() {
    try {
      const response = await api.get(API_ENDPOINTS.AVAILABLE_DRILLS);
      return {
        success: true,
        drills: response.data.drills || [],
        total: response.data.total_count || 0
      };
    } catch (error) {
      console.error('Failed to fetch drills:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to load drills'
      };
    }
  }

  async getDrillInfo(drillType) {
    try {
      const response = await api.get(`/drill/info/${drillType}`);
      return {
        success: true,
        drill: response.data
      };
    } catch (error) {
      console.error('Failed to fetch drill info:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to load drill information'
      };
    }
  }

  async getDrillBenchmark(drillType) {
    try {
      const response = await api.get(`/drill/benchmark/${drillType}`);
      return {
        success: true,
        benchmark: response.data
      };
    } catch (error) {
      console.error('Failed to fetch drill benchmark:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to load benchmark data'
      };
    }
  }

  async startDrillAnalysis(drillType, videoFile) {
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: videoFile.uri,
        type: 'video/mp4',
        name: 'drill_video.mp4'
      });
      formData.append('drill_type', drillType);

      const response = await api.post(API_ENDPOINTS.ANALYZE_DRILL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return {
        success: true,
        analysisId: response.data.analysis_id,
        drillType: response.data.drill_type
      };
    } catch (error) {
      console.error('Failed to start drill analysis:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to start analysis'
      };
    }
  }

  async getAnalysisStatus(analysisId) {
    try {
      const response = await api.get(API_ENDPOINTS.DRILL_STATUS(analysisId));
      return {
        success: true,
        status: response.data
      };
    } catch (error) {
      console.error('Failed to get analysis status:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to get analysis status'
      };
    }
  }

  async getAnalysisResults(analysisId) {
    try {
      const response = await api.get(API_ENDPOINTS.DRILL_RESULTS(analysisId));
      return {
        success: true,
        results: response.data
      };
    } catch (error) {
      console.error('Failed to get analysis results:', error);
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to get results'
      };
    }
  }
}

export default new DrillService();
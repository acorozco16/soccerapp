'use client'

import { useState } from 'react'
import { Trophy, Clock, TrendingUp, Target, Image, RotateCcw } from 'lucide-react'
import { AnalysisResults } from '@/lib/types'
import { api } from '@/lib/api'

interface ResultsDisplayProps {
  results: AnalysisResults
  videoId: string
  onReset: () => void
}

export default function ResultsDisplay({ results, videoId, onReset }: ResultsDisplayProps) {
  const [selectedFrame, setSelectedFrame] = useState<string | null>(null)

  const getConfidenceLabel = (score: number) => {
    if (score >= 0.8) return { label: 'High Confidence', color: 'text-green-600' }
    if (score >= 0.6) return { label: 'Good Confidence', color: 'text-yellow-600' }
    return { label: 'Low Confidence', color: 'text-orange-600' }
  }

  const confidence = getConfidenceLabel(results.confidence_score)

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-green-100 rounded-full mb-4">
            <Trophy className="w-10 h-10 text-green-600" />
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Analysis Complete!
          </h2>
          <p className="text-gray-600">
            Here are your soccer touch statistics
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-700 font-medium mb-1">Total Touches</p>
                <p className="text-4xl font-bold text-green-900">
                  {results.total_ball_touches}
                </p>
              </div>
              <div className="text-5xl">⚽</div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-blue-700 font-medium mb-1">Touches/Minute</p>
                <p className="text-4xl font-bold text-blue-900">
                  {results.touches_per_minute}
                </p>
              </div>
              <TrendingUp className="w-12 h-12 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-1">
              <Clock className="w-5 h-5 text-gray-600" />
              <p className="text-sm text-gray-600">Video Duration</p>
            </div>
            <p className="text-xl font-semibold">{results.video_duration.toFixed(1)}s</p>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-1">
              <Target className="w-5 h-5 text-gray-600" />
              <p className="text-sm text-gray-600">Confidence Score</p>
            </div>
            <p className={`text-xl font-semibold ${confidence.color}`}>
              {(results.confidence_score * 100).toFixed(0)}%
            </p>
            <p className={`text-xs ${confidence.color}`}>{confidence.label}</p>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-1">
              <Clock className="w-5 h-5 text-gray-600" />
              <p className="text-sm text-gray-600">Processing Time</p>
            </div>
            <p className="text-xl font-semibold">{results.processing_time.toFixed(1)}s</p>
          </div>
        </div>

        {results.debug_frames && results.debug_frames.length > 0 && (
          <div className="border-t pt-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Image className="w-5 h-5" />
              <span>Analysis Frames</span>
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {results.debug_frames.map((frame) => (
                <button
                  key={frame}
                  onClick={() => setSelectedFrame(frame)}
                  className="relative group overflow-hidden rounded-lg border-2 border-gray-200 hover:border-green-500 transition"
                >
                  <img
                    src={api.getFrameUrl(videoId, frame)}
                    alt={frame}
                    className="w-full h-32 object-cover"
                  />
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                    <span className="text-white text-sm font-medium">
                      View Frame
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="flex justify-center mt-8">
          <button
            onClick={onReset}
            className="inline-flex items-center space-x-2 bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition"
          >
            <RotateCcw className="w-5 h-5" />
            <span>Analyze Another Video</span>
          </button>
        </div>
      </div>

      {selectedFrame && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedFrame(null)}
        >
          <div className="max-w-4xl max-h-[90vh] relative">
            <img
              src={api.getFrameUrl(videoId, selectedFrame)}
              alt="Analysis frame"
              className="w-full h-full object-contain rounded-lg"
            />
            <button
              className="absolute top-4 right-4 bg-white/10 backdrop-blur text-white p-2 rounded-full hover:bg-white/20 transition"
              onClick={(e) => {
                e.stopPropagation()
                setSelectedFrame(null)
              }}
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
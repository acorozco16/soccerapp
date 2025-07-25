'use client'

import { useEffect, useState } from 'react'
import { Loader2, CheckCircle } from 'lucide-react'
import { api } from '@/lib/api'
import { AnalysisResults } from '@/lib/types'

interface ProcessingStatusProps {
  videoId: string
  onComplete: (results: AnalysisResults) => void
  onError: (error: string) => void
}

const processingStages = [
  "Uploading video...",
  "Analyzing video frames...",
  "Detecting ball position...",
  "Tracking player movements...",
  "Counting ball touches...",
  "Generating results..."
]

export default function ProcessingStatus({ videoId, onComplete, onError }: ProcessingStatusProps) {
  const [stage, setStage] = useState(0)
  const [processingTime, setProcessingTime] = useState(0)

  useEffect(() => {
    let pollInterval: NodeJS.Timeout
    let stageInterval: NodeJS.Timeout
    let startTime = Date.now()

    const checkStatus = async () => {
      try {
        const status = await api.getStatus(videoId)
        
        // Update processing time
        setProcessingTime(Math.floor((Date.now() - startTime) / 1000))

        if (status.status === 'completed') {
          clearInterval(pollInterval)
          clearInterval(stageInterval)
          
          // Get results
          const results = await api.getResults(videoId)
          onComplete(results)
        } else if (status.status === 'error') {
          clearInterval(pollInterval)
          clearInterval(stageInterval)
          onError(status.error || 'Processing failed')
        }
      } catch (err: any) {
        clearInterval(pollInterval)
        clearInterval(stageInterval)
        onError(err.response?.data?.detail || 'Failed to check status')
      }
    }

    // Start polling
    pollInterval = setInterval(checkStatus, 2000)
    checkStatus() // Initial check

    // Animate through stages
    stageInterval = setInterval(() => {
      setStage((prev) => (prev + 1) % processingStages.length)
    }, 3000)

    return () => {
      clearInterval(pollInterval)
      clearInterval(stageInterval)
    }
  }, [videoId, onComplete, onError])

  return (
    <div className="bg-white rounded-lg shadow-lg p-8">
      <div className="text-center">
        <div className="relative inline-flex">
          <Loader2 className="w-16 h-16 text-green-600 animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-2xl">⚽</span>
          </div>
        </div>
        
        <h2 className="text-2xl font-semibold mt-6 mb-2">Processing Your Video</h2>
        <p className="text-gray-600 mb-6">{processingStages[stage]}</p>

        <div className="max-w-md mx-auto">
          <div className="flex justify-between text-sm text-gray-500 mb-2">
            <span>Progress</span>
            <span>{processingTime}s</span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div className="bg-green-600 h-full rounded-full transition-all duration-1000 relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
            </div>
          </div>
        </div>

        <div className="mt-8 space-y-2">
          {processingStages.map((stageText, index) => (
            <div
              key={index}
              className={`flex items-center space-x-2 text-sm ${
                index <= stage ? 'text-green-600' : 'text-gray-400'
              }`}
            >
              {index < stage ? (
                <CheckCircle className="w-4 h-4" />
              ) : index === stage ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <div className="w-4 h-4 rounded-full border border-current" />
              )}
              <span>{stageText}</span>
            </div>
          ))}
        </div>

        <p className="text-sm text-gray-500 mt-6">
          This usually takes 1-3 minutes depending on video length
        </p>
      </div>
    </div>
  )
}
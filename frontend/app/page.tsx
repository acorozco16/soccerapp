'use client'

import { useState } from 'react'
import VideoUpload from '@/components/VideoUpload'
import ProcessingStatus from '@/components/ProcessingStatus'
import ResultsDisplay from '@/components/ResultsDisplay'
import { VideoState } from '@/lib/types'

export default function Home() {
  const [videoState, setVideoState] = useState<VideoState>({ status: 'idle' })

  return (
    <main className="min-h-screen bg-gradient-to-b from-green-50 to-white">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            ⚽ Soccer Touch Counter
          </h1>
          <p className="text-lg text-gray-600">
            Analyze your soccer videos to count ball touches automatically
          </p>
        </header>

        {videoState.status === 'idle' && (
          <VideoUpload onVideoUploaded={(videoId) => 
            setVideoState({ status: 'processing', videoId })
          } />
        )}

        {videoState.status === 'processing' && videoState.videoId && (
          <ProcessingStatus 
            videoId={videoState.videoId}
            onComplete={(results) => 
              setVideoState({ status: 'completed', videoId: videoState.videoId, results })
            }
            onError={(error) => 
              setVideoState({ status: 'error', error })
            }
          />
        )}

        {videoState.status === 'completed' && videoState.results && (
          <ResultsDisplay 
            results={videoState.results}
            videoId={videoState.videoId!}
            onReset={() => setVideoState({ status: 'idle' })}
          />
        )}

        {videoState.status === 'error' && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <h2 className="text-xl font-semibold text-red-800 mb-2">
              ❌ Processing Error
            </h2>
            <p className="text-red-600 mb-4">{videoState.error}</p>
            <button
              onClick={() => setVideoState({ status: 'idle' })}
              className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition"
            >
              Try Again
            </button>
          </div>
        )}

        <footer className="mt-16 text-center text-gray-500 text-sm">
          <p>Record 30-60 seconds of soccer practice</p>
          <p>Works best with good lighting and clear ball visibility</p>
        </footer>
      </div>
    </main>
  )
}
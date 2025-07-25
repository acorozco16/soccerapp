'use client'

import { useState, useRef } from 'react'
import { Upload, Video, AlertCircle } from 'lucide-react'
import { api } from '@/lib/api'

interface VideoUploadProps {
  onVideoUploaded: (videoId: string) => void
}

export default function VideoUpload({ onVideoUploaded }: VideoUploadProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-m4v']
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|mov)$/i)) {
      setError('Please select an MP4 or MOV video file')
      return
    }

    // Validate file size (100MB limit)
    if (file.size > 100 * 1024 * 1024) {
      setError('File too large. Maximum size is 100MB')
      return
    }

    setError(null)
    setSelectedFile(file)
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setIsUploading(true)
    setError(null)
    setUploadProgress(0)

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 10
        })
      }, 200)

      const result = await api.uploadVideo(selectedFile)
      
      clearInterval(progressInterval)
      setUploadProgress(100)
      
      setTimeout(() => {
        onVideoUploaded(result.video_id)
      }, 500)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.')
      setIsUploading(false)
      setUploadProgress(0)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 md:p-8">
      <div className="text-center mb-6">
        <Video className="w-16 h-16 text-green-600 mx-auto mb-4" />
        <h2 className="text-2xl font-semibold mb-2">Upload Your Soccer Video</h2>
        <p className="text-gray-600">
          Record 30-60 seconds using your phone's camera, then upload here
        </p>
      </div>

      <div className="space-y-4">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-green-500 transition-colors">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/quicktime,.mp4,.mov"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isUploading}
          />

          {!selectedFile ? (
            <button
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center space-x-2 text-gray-600 hover:text-green-600 transition"
              disabled={isUploading}
            >
              <Upload className="w-8 h-8" />
              <span className="text-lg">Choose Video File</span>
            </button>
          ) : (
            <div className="space-y-3">
              <p className="text-green-600 font-medium">
                ✓ {selectedFile.name}
              </p>
              <p className="text-sm text-gray-500">
                {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
              </p>
              {!isUploading && (
                <button
                  onClick={() => {
                    setSelectedFile(null)
                    if (fileInputRef.current) {
                      fileInputRef.current.value = ''
                    }
                  }}
                  className="text-sm text-gray-500 hover:text-red-600"
                >
                  Choose different file
                </button>
              )}
            </div>
          )}
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-2">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}

        {isUploading && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-gray-600">
              <span>Uploading...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {selectedFile && !isUploading && (
          <button
            onClick={handleUpload}
            className="w-full bg-green-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-green-700 transition duration-200 flex items-center justify-center space-x-2"
          >
            <Upload className="w-5 h-5" />
            <span>Upload and Analyze</span>
          </button>
        )}
      </div>

      <div className="mt-6 bg-blue-50 rounded-lg p-4">
        <h3 className="font-medium text-blue-900 mb-2">Recording Tips:</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Use landscape mode for best results</li>
          <li>• Ensure good lighting conditions</li>
          <li>• Keep the ball visible in frame</li>
          <li>• Avoid excessive camera movement</li>
        </ul>
      </div>
    </div>
  )
}
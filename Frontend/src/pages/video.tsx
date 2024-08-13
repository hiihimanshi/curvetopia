import React from 'react';

const Video: React.FC = () => {
  return (
    <div className="video-container">
      <iframe
        className="video-player"
        src="https://www.youtube.com/embed/YOUR_VIDEO_ID"
        title="YouTube Video Player"
        allowFullScreen
      ></iframe>
    </div>
  );
};

export default Video;
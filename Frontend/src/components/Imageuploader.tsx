// frontend/src/components/ImageUploader.tsx
import React, { useState } from 'react';

const ImageUploader: React.FC = () => {
    const [image, setImage] = useState<File | null>(null);

    // Function to handle image upload
    const onImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if ( e.target.files && e.target.files.length > 0) {
            setImage(e.target.files[0]);
        }
    };

    // Function to handle feature button clicks
    const handleFeatureClick = (feature: string) => {
        if (!image) {
            console.error('No image file provided');
            return;
        }

        const formData = new FormData();
        formData.append('file', image);
        formData.append('feature', feature);

        // Send the image and selected feature to the Flask backend
        fetch('http://localhost:5000/process_image', {
            method: 'POST',
            body: formData,
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then((blob) => {
            const url = URL.createObjectURL(blob);
            // Open the processed image in a new window
            window.open(url);
        })
        .catch((error) => console.error('Error:', error));
    };

    return (
        <div>
            <input type="file" onChange={onImageChange} />
            <button onClick={() => handleFeatureClick('Symmetry')}>Symmetry</button>
            <button onClick={() => handleFeatureClick('Regularized')}>Regularized</button>
            <button onClick={() => handleFeatureClick('Plannerized')}>Plannerized</button>
        </div>
    );
};

export default ImageUploader;

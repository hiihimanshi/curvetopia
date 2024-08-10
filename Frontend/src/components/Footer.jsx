
/* eslint-disable no-unused-vars */
import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-gray-400 py-8 mt-12">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center">
          <p>&copy; 2024 Image Enhancement Hub. All rights reserved.</p>
          <ul className="flex space-x-4">
            <li><a href="#" className="hover:text-white">Privacy Policy</a></li>
            <li><a href="#" className="hover:text-white">Terms of Service</a></li>
            <li><a href="#" className="hover:text-white">Support</a></li>
          </ul>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

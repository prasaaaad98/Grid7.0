"use client"

import { useState, useEffect } from "react"
import { Star, Heart } from "lucide-react"

interface Product {
  id: number
  title: string
  brand: string
  category: string
  price: number
  retail_price: number
  images: string[]
  rating: number
  description: string
  isSponsored?: boolean
}

interface ResultsGridProps {
  query: string
  filters: any
  sort: string
  userLat: number | null
  userLon: number | null
}

export default function ResultsGrid({ query, filters, sort, userLat, userLon }: ResultsGridProps) {
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(false)
  
  console.log('ResultsGrid render - sort:', sort, 'query:', query)

  useEffect(() => {
    if (!query.trim()) {
      setProducts([])
      return
    }

    setLoading(true)

    // Real backend call
    const searchParams = new URLSearchParams({
      q: query,
      min_price: filters.priceMin.toString(),
      max_price: filters.priceMax.toString(),
      min_rating: filters.rating.toString(),
      brand: filters.brands.join(','),
      category: filters.categories.join(','),
      sort: sort
    })

    console.log('Search params:', searchParams.toString())
    fetch(`http://localhost:8000/search?${searchParams}`)
      .then(res => res.json())
      .then(data => {
        console.log('Received data:', data.results?.slice(0, 3).map((item: any) => ({ title: item.title, price: item.price })))
        // Handle new backend response format with results, total_hits, and facets
        const products = data.results || data // Fallback to old format if needed
        const sponsoredId = data.sponsored_id // Get sponsored product ID
        
        // Transform backend data to match frontend Product interface
        const transformedProducts: Product[] = products.map((item: any, index: number) => ({
          id: item.id || index + 1,
          title: item.title,
          brand: item.brand,
          category: item.category,
          price: item.price,
          retail_price: item.retail_price,
          images: item.images,
          rating: item.rating,
          description: item.description,
          isSponsored: item.id === sponsoredId,
        }))
        console.log('Setting products:', transformedProducts.slice(0, 3).map(p => ({ title: p.title, price: p.price })))
        setProducts(transformedProducts)
      })
      .catch(error => {
        console.error('Error fetching search results:', error)
        setProducts([]) // Just set empty array on error
      })
      .finally(() => setLoading(false))
  }, [query, filters, sort, userLat, userLon])

  const addToCart = (product: Product) => {
    const cart = JSON.parse(localStorage.getItem("cart") || "[]")
    const existingItemIndex = cart.findIndex((item: any) => item.id === product.id)

    if (existingItemIndex > -1) {
      // Item exists, increase quantity
      cart[existingItemIndex].quantity += 1
    } else {
      // New item, add to cart
      cart.push({
        id: product.id,
        title: product.title,
        price: Math.round(product.price),
        quantity: 1,
      })
    }

    localStorage.setItem("cart", JSON.stringify(cart))

    // Trigger storage event for cart count update
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "cart",
        newValue: JSON.stringify(cart),
      }),
    )

    alert(`${product.title} added to cart!`)
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-white p-4 rounded shadow animate-pulse">
              <div className="bg-gray-200 h-48 rounded mb-4"></div>
              <div className="bg-gray-200 h-4 rounded mb-2"></div>
              <div className="bg-gray-200 h-4 rounded w-3/4"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!query) {
    return (
      <div className="p-8 text-center text-gray-500">
        <p>Search for products to see results</p>
      </div>
    )
  }

  return (
    <div className="p-2 sm:p-4">
      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 sm:gap-4">
        {products.map((product, index) => {
          return (
            <div key={`${product.id}-${index}`} className="bg-white rounded shadow hover:shadow-lg transition-shadow p-2 sm:p-4">
              <div className="relative mb-2 sm:mb-4">
                <img
                  src={product.images && product.images.length > 0 ? product.images[0] : "/placeholder.svg"}
                  alt={product.title}
                  loading="lazy"
                  className="w-full h-32 sm:h-48 object-cover rounded"
                />
                <button className="absolute top-1 right-1 sm:top-2 sm:right-2 p-1 bg-white rounded-full shadow">
                  <Heart size={12} className="sm:w-4 sm:h-4 text-gray-400" />
                </button>
              </div>

              <h3 className="font-medium text-xs sm:text-sm mb-1 sm:mb-2 line-clamp-2">{product.title}</h3>

              <div className="flex items-center mb-1 sm:mb-2">
                <div className="flex items-center bg-green-500 text-white px-1 py-0.5 rounded text-xs">
                  <span className="text-xs">{product.rating}</span>
                  <Star size={8} className="sm:w-2.5 sm:h-2.5 ml-1 fill-current" />
                </div>
                {product.isSponsored && (
                  <span className="text-xs text-gray-500 ml-2">*sponsored</span>
                )}
              </div>

              <div className="mb-1 sm:mb-2">
                <span className="text-sm sm:text-lg font-semibold">
                  ₹{Math.round(product.price).toLocaleString()}
                </span>
                {product.retail_price > product.price && (
                  <span className="text-xs sm:text-sm text-gray-500 line-through ml-1 sm:ml-2">
                    ₹{product.retail_price.toLocaleString()}
                  </span>
                )}
              </div>

              <div className="space-y-1 sm:space-y-2">
                <button
                  onClick={() => addToCart(product)}
                  className="w-full bg-[#ff9f00] hover:bg-[#e68900] text-white py-1 sm:py-2 px-2 sm:px-4 rounded text-xs sm:text-sm font-medium transition-colors"
                >
                  Add to Cart
                </button>
                <button className="w-full bg-[#fb641b] hover:bg-[#e55a1b] text-white py-1 sm:py-2 px-2 sm:px-4 rounded text-xs sm:text-sm font-medium transition-colors">
                  Buy Now
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {products.length === 0 && query && (
        <div className="text-center py-8 text-gray-500">
          <p className="text-sm sm:text-base">No products found for "{query}"</p>
          <p className="text-xs sm:text-sm mt-2">Try searching with different keywords</p>
        </div>
      )}
    </div>
  )
}
